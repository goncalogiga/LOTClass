from transformers import PreTrainedModel
# =================================================================================
from transformers import FlaubertModel
from transformers.models.flaubert.modeling_flaubert import FlaubertWithLMHeadModel
#from transformers import CamembertModel
#from transformers.models.camembert.modeling_camembert import CamembertForMaskedLM
# =================================================================================
from torch import nn
import sys


class FlaubertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = FlaubertWithLMHeadModel(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class LOTClassModel(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = FlaubertModel(config)
        #self.bert = CamembertModel(config, add_pooling_layer=False)
        self.cls = FlaubertOnlyMLMHead(config)
        #self.cls = CamembertForMaskedLM(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        # MLM head is not trained
        for param in self.cls.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, pred_mode, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None):
        bert_outputs = self.bert(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 inputs_embeds=inputs_embeds)
        last_hidden_states = bert_outputs[0]
        if pred_mode == "classification":
            trans_states = self.dense(last_hidden_states)
            trans_states = self.activation(trans_states)
            trans_states = self.dropout(trans_states)
            logits = self.classifier(trans_states)
        elif pred_mode == "mlm":
            print("Last hidden states size:", last_hidden_states.size())
            logits = self.cls(last_hidden_states)
        else:
            sys.exit("Wrong pred_mode!")
        print(f"Model output size: {logits.size()}")
        print(f"Model output type: {type(logits)}")
        return logits
