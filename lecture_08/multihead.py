import torch

batch_size = 3
emb_size = 16
enc_output_length = 7
dec_input_length = 11

x_enc = torch.normal(0, 1, (batch_size, enc_output_length, emb_size))
x_dec = torch.normal(0, 1, (batch_size, dec_input_length, emb_size))

cross_attn = torch.nn.MultiheadAttention(emb_size, num_head=2, batch_first=True)
c, weights = cross_attn(x_dec, x_enc, x_enc)

# masked_attn = torch.nn.MultiheadAttention(emb_size, num_head=2, batch_first=True)
# mask = torch.triu(torch.ones(dec_input_length, dec_input_length) * float('-inf'), diagonal=1)
# c, weights = masked_attn(x_dec, x_enc, x_enc)
#
# self_attn = torch.nn.MultiheadAttention(emb_size, num_head=2, batch_first=True)
# c, weights = self_attn(x_dec, x_enc, x_enc)