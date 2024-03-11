import os
import torch

BOS_IDX = 1
EOS_IDX = 69

class Predictor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config.device)

        self.model = self.get_model()
        self.model.to(self.device)

        self.logs_dir = os.path.join(self.config.root_dir, self.config.experiment_name)
        path = os.path.join(self.logs_dir, "best_checkpoint.pth")
        self.model.load_state_dict(torch.load(path)["state_dict"])

    def get_model(self):
        if self.config.model_name == "seq2seq_transformer":
            from model.seq2seq import Model
            model = Model(num_encoder_layers=self.config.num_encoder_layers,
                          num_decoder_layers=self.config.num_decoder_layers,
                          emb_size=self.config.embedding_size,
                          nhead=self.config.nhead,
                          src_vocab_size=self.config.src_vocab_size,
                          tgt_vocab_size=self.config.tgt_vocab_size,
                          input_emb_size=self.config.input_emb_size,
                          max_input_points=self.config.max_input_points,
                          )
        else:
            raise NotImplementedError
        
        return model
    
    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)
        dim = 1
        
        memory = self.model.encode(src, src_mask)
        memory = memory.to(self.device)
        dim = 1
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len-1):
            
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(1), self.device).type(torch.bool)).to(self.device)
            out = self.model.decode(ys, memory, tgt_mask)
            # out = out.transpose(0, 1)
            prob = self.model.generator(out[:, -1])
            
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            print(next_word)
            
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=dim)
            if next_word == EOS_IDX:
                break
        return ys


    def predict(self, test_example, raw_tokens=False):
        self.model.eval()

        src = test_example
        num_tokens = src.shape[1]
        
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, src_mask, max_len=256, start_symbol=BOS_IDX).flatten()
        
        return tgt_tokens