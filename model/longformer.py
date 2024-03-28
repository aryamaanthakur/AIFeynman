''' Longformer Encoder Decoder Model'''
import torch
import transformers
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from transformers import LEDForConditionalGeneration, LEDConfig


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        configuration = LEDConfig(vocab_size = self.config.vocab_size,
                                  encoder_layers = self.config.num_encoder_layers,
                                  decoder_layers = self.config.num_decoder_layers,
                                  d_model = self.config.embedding_size,
                                  dropout = self.config.dropout,
                                  decoder_ffn_dim = self.config.hidden_dim,
                                  encoder_ffn_dim = self.config.hidden_dim,
                                  encoder_attention_heads = self.config.nhead,
                                  decoder_attention_heads = self.config.nhead,
                                  max_encoder_position_embeddings = self.config.maximum_sequence_length,
                                  max_decoder_position_embeddings = self.config.maximum_sequence_length
                                 )
        
        self.LED_pretrained = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
        self.LED = LEDForConditionalGeneration(config = configuration)
        
        if self.config.pretrain:
            self.LED.load_state_dict(self.LED_pretrained.state_dict()) # Keep the hyperparameters exact to the pretrained model
            print("==> Loaded the pre-trained weights")
            
        
    def forward(self, input_ids, decoder_input_ids):
        output = self.LED(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        return output
