from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoConfig, PreTrainedModel, WhisperModel, AutoModel
import torch.nn as nn 
import torch

class EmotionClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels=5, dropout=0.2):
        super().__init__()
        self.pool_norm = nn.LayerNorm(hidden_size)
        self.pre_dropout = nn.Dropout(dropout)

        mid1 = max(hidden_size // 2, num_labels * 4)
        mid2 = max(hidden_size // 4, num_labels * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, mid1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(mid1),
            nn.Linear(mid1, mid2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(mid2),
            nn.Linear(mid2, num_labels),
        )

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1, keepdim=True)
            masked = hidden_states * attention_mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / lengths
        else:
            pooled = hidden_states.mean(dim=1)
        x = self.pool_norm(pooled)
        x = self.pre_dropout(x)
        logits = self.classifier(x)
        return logits

class WhisperForEmotionClassification(PreTrainedModel):
    config_class = AutoConfig

    def __init__(
        self, config, model_name, num_labels=5, dropout=0.2
    ):
        super().__init__(config)
        self.encoder = WhisperModel.from_pretrained(model_name).encoder
        hidden_size = config.hidden_size
        self.classifier = EmotionClassifier(
            hidden_size, num_labels=num_labels, dropout=dropout
        )
        self.post_init()

    def forward(self, input_features, attention_mask=None, labels=None):
        encoder_output = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = encoder_output.last_hidden_state
        logits = self.classifier(hidden_states, attention_mask=attention_mask)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
    
class ModelForEmotionClassification(PreTrainedModel):
    config_class = AutoConfig

    def __init__(
        self, config, model_name, num_labels=5, dropout=0.2
    ):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True).model.encoder
        hidden_size = config.encoder['d_model']
        self.classifier = EmotionClassifier(
            hidden_size, num_labels=num_labels, dropout=dropout
        )
        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor,
        input_lengths: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> SequenceClassifierOutput:
        encoded, out_lens = self.encoder(input_features, input_lengths)
        hidden_states = encoded.transpose(1, 2)

        if attention_mask is None:
            max_t = hidden_states.size(1)
            attention_mask = (
                torch.arange(max_t, device=out_lens.device)
                .unsqueeze(0)
                .lt(out_lens.unsqueeze(1))
                .long()
            )

        logits = self.classifier(hidden_states, attention_mask=attention_mask)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)