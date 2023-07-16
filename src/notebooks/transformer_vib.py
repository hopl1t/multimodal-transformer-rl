import torch
from transformers import BertForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoModel, AutoTokenizer
from torch.nn import Identity
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
import pickle
import dill
from textattack.models.wrappers.huggingface_model_wrapper import HuggingFaceModelWrapper
from transformers import PreTrainedModel
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions.normal import Normal
import transformers
from transformers.modeling_outputs import SequenceClassifierOutput

from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import BAEGarg2019
from textattack.goal_functions import UntargetedClassification
from textattack.attack_results import SuccessfulAttackResult
from textattack import Attacker, AttackArgs
from textattack.metrics.attack_metrics import (
    AttackQueries,
    AttackSuccessRate,
    WordsPerturbed,
)


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


class TransformerVIB(nn.Module):
    """
    Classifier with stochastic layer and KL regularization
    """

    def __init__(self, hidden_size, output_size, device):
        super(TransformerVIB, self).__init__()
        self.device = device
        self.description = 'Vanilla IB VAE as per the paper'
        self.hidden_size = hidden_size
        self.k = hidden_size // 2
        self.output_size = output_size
        self.train_loss = []
        self.test_loss = []

        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Linear(self.k, output_size)
        # These are cheats to make 'drill' save everythung we need in one pickle
        self.softplus = F.softplus
        self.normal = torch.normal
        self.Normal = Normal

        # Xavier initialization
        for _, module in self._modules.items():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(
                    module.weight, gain=nn.init.calculate_gain('relu'))
                module.bias.data.zero_()
                continue
            for layer in module:
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(
                        layer.weight, gain=nn.init.calculate_gain('relu'))
                    layer.bias.data.zero_()

    def reparametrize(self, mu, std, device):
        """
        Performs reparameterization trick z = mu + epsilon * std
        Where epsilon~N(0,1)
        """
        mu = mu.expand(1, *mu.size())
        std = std.expand(1, *std.size())
        eps = self.normal(0, 1, size=std.size()).to(device)
        return mu + eps * std

    def forward(self, x):
        z_params = self.encoder(x)
        mu = z_params[:, :self.k]
#         std = torch.nn.functional.softplus(z_params[:, self.k:] - 1, beta=1)
        std = self.softplus(z_params[:, self.k:] - 1, beta=1)
        if self.training:
            z = self.reparametrize(mu, std, self.device)
        else:
            z = mu.clone().unsqueeze(0)
        n = self.Normal(mu, std)
        # These may be positive as this is a PDF
        log_probs = n.log_prob(z.squeeze(0))

        logits = self.classifier(z)
        return (mu, std), log_probs, logits


class TransformerHybridModel(nn.Module):
    """
    Head is a pretrained model, classifier is VIB
    fc_name should be 'fc2' for inception-v3 (imagenet) and mnist-cnn, '_fc' for efficient-net (CIFAR)
    """

    def __init__(self, base_model, vib_model, device, fc_name, return_only_logits=False):
        super(TransformerHybridModel, self).__init__()
        self.device = device
        self.base_model = base_model
        setattr(self.base_model, fc_name, torch.nn.Identity())
        self.vib_model = vib_model
        self.train_loss = []
        self.test_loss = []
        self.freeze_base()
        self.return_only_logits = return_only_logits

    def set_return_only_logits(self, bool_value):
        self.return_only_logits = bool_value

    def freeze_base(self):
        # Freeze the weights of the inception_model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        # Freeze the weights of the inception_model
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, **kwargs):
        # This is not really logits, only called that way cause we've changed the final layer to identity
        encoded = self.base_model(kwargs['input_ids']).logits
        (mu, std), log_probs, logits = self.vib_model(encoded)
        if self.return_only_logits:
            return logits.squeeze(0)
        else:
            return ((mu, std), log_probs, logits)


class TransformerAdaptor(transformers.PreTrainedModel):
    """
    Adapts between a TransformerHybridModel to a HuggingFaceModelWrapper
    """

    def __init__(self, hybrid_model):
        super(TransformerAdaptor, self).__init__(
            hybrid_model.base_model.config)
        self.hybrid_model = hybrid_model
        # Cheat to overload drill pickle
        self.SequenceClassifierOutput = SequenceClassifierOutput

    def forward(self, **kwargs):
        if self.hybrid_model.return_only_logits:
            logits = self.hybrid_model(**kwargs)
        else:
            ((_, _), _, logits) = self.hybrid_model(**kwargs)
        return self.SequenceClassifierOutput(logits=logits[0])


def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)


def bae_attack(hybrid_model):
    """
    Performs a BAE-R untargeted blackbox attack using the textattack API
    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model_adaptor = TransformerAdaptor(hybrid_model)
    model_wrapper = HuggingFaceModelWrapper(model_adaptor, tokenizer)

    dataset = HuggingFaceDataset("yelp_polarity", None, "test")

    attack = BAEGarg2019.build(model_wrapper)
    attack_args = AttackArgs(
        num_examples=250,
        disable_stdout=True,
        silent=True,
        enable_advance_metrics=True
    )

    attacker = Attacker(attack, dataset, attack_args)
    results_iterable = attacker.attack_dataset()
    attack_log_manager = attacker.attack_log_manager
    attack_log_manager.log_summary()
    attack_success_stats = AttackSuccessRate().calculate(attack_log_manager.results)
    words_perturbed_stats = WordsPerturbed().calculate(attack_log_manager.results)
    attack_query_stats = AttackQueries().calculate(attack_log_manager.results)
    acc_under_attack = str(attack_success_stats["attack_accuracy_perc"])
    avg_pertrubed_words_prct = str(words_perturbed_stats["avg_word_perturbed_perc"])
    original_acc = attack_success_stats["original_accuracy"]

    return original_acc, acc_under_attack, avg_pertrubed_words_prct