#评估和训练在forward难以分离

class Seq2Seq(nn.Module):

    def __init__(self,
                 vocab_size,
                 word_embed_dim,
                 hidden_dim,
                 num_layers=1,
                 pos_embed_dim=None,
                 pos_size=None,
                 ner_embed_dim=None,
                 ner_size=None,
                 pre_word_embeds=None,
                 bi=True):
        super(Seq2Seq, self).__init__()

        # 它的最后输出的隐层尺寸一定是和vocab_size一致的，这样才能找到哪个可能性最高

        # Encoder Part #################################################
        self.pos_embed_dim = pos_embed_dim
        self.ner_embed_dim = ner_embed_dim
        self.hidden_dim = hidden_dim

        # EMBEDDING
        self.word_embeds = nn.Embedding(vocab_size, word_embed_dim)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(
                torch.FloatTensor(pre_word_embeds))
            self.word_embeds.weight.requires_grad = False
        else:
            self.pre_word_embeds = False

        self.e_embed_dim = word_embed_dim

        if pos_embed_dim is not None and pos_size is not None:
            self.pos_embeds = nn.Embedding(pos_size, pos_embed_dim)
            self.e_embed_dim += pos_embed_dim

        if ner_embed_dim is not None and ner_size is not None:
            self.ner_embeds = nn.Embedding(ner_size, ner_embed_dim)
            self.e_embed_dim += ner_embed_dim

        # GRU
        self.nn_word = nn.GRU(self.e_embed_dim, self.e_hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bi)
        e_hidden_dim_total = self.e_hidden_dim  # bidirectional

        if pos_embed_dim is not None and pos_size is not None:
            self.nn_pos = nn.GRU(self.e_embed_dim, self.e_hidden_dim,
                                 num_layers=num_layers,
                                 bidirectional=bi)
            e_hidden_dim_total += self.e_hidden_dim

        if ner_embed_dim is not None and ner_size is not None:
            self.nn_ner = nn.GRU(self.e_embed_dim, self.e_hidden_dim,
                                 num_layers=num_layers,
                                 bidirectional=bi)
            e_hidden_dim_total += self.e_hidden_dim

        # Decoder Part #################################################
        self.gru = nn.GRU(self.a_hidden_dim, e_hidden_dim_total,
                          num_layers=num_layers, bidirectional=bi)

        self.dropout = nn.Dropout(dropout_rate)

        self.out_word = nn.Linear(self.d_hidden_dim * 2, vocab_size)
        self.out_ner = nn.Linear(self.d_hidden_dim * 2, ner_size)
        self.out_pos = nn.Linear(self.d_hidden_dim * 2, pos_size)

        # Attention part #################################################
        self.attn = nn.Linear(
            self.embed_dim + self.e_hidden_dim, self.max_length)
        self.attn_combine = nn.Linear(
            self.embed_dim + self.e_hidden_dim, self.a_hidden_dim)

    def _get_feature(self, words, ners=None, poss=None):
        embeds = self.word_embeds(words).view(
            len(words), 1, -1)
        # embeds: seqlen by 1 by word_embed_dim
        if self.pos_embed_dim is not None and poss is not None:
            pos_embeds = self.word_embeds(poss).view(
                len(poss), 1, -1)
            embeds = torch.cat((embeds, pos_embeds), 2)
            # embeds: seqlen by 1 by word_embed_dim+pos_embed_dim

        if self.ner_embed_dim is not None and ners is not None:
            ner_embeds = self.word_embeds(ners).view(
                len(ners), 1, -1)
            embeds = torch.cat((embeds, ner_embeds), 2)
            # embeds: seqlen by 1 by word_embed_dim+pos_embed_dim+ner_embed_dim

        return embeds

    def attention(self, embedded, hidden):
        # embeds: 1 by 1 by self.e_embed_dim
        # hidden: 1 by 1 by self.e_hidden_dim * 3
        # att:    e_embed_dim + e_hidden_dim * 3 -> max_length
        # attn_weights： 1 by self.max_length
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        # attn_weights: 1 by self.max_length
        # self.encoder_output: self.max_length by self.e_hidden_dim * 3
        # attn_applied: 1 by 1 by self.e_hidden_dim * 3
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 self.encoder_output.unsqueeze(0))

        # attn_applied: 1 by 1 by self.e_hidden_dim * 3
        # embedded: 1 by 1 by self.e_embed_dim
        # output: 1 by 1 by (self.e_embed_dim + self.e_hidden_dim * 3)
        output = torch.cat((embedded[0], attn_applied[0]), 1)

        # attn_combine 线性映射 self.e_embed_dim + self.e_hidden_dim * 3 -> self.a_hidden_dim 这里可以做压缩
        # output size 1 by 1 by a_hidden_dim
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        # output size 1 by 1 by a_hidden_dim
        # attn_weights: 1 by self.max_length
        return output, attn_weights

    def encode(self, ):

        length =

        # Embedding
        # embeds: seqlen by 1 by word_embed_dim+pos_embed_dim+ner_embed_dim
        # (e_embed_dim)
        embeds = self._get_feature(words, ners, poss)

        # GRU
        output_word, hidden_word = self.nn_word(embeds)
        output, hidden = output_word, hidden_word
        # output: seqlen by 1 by self.e_embed_dim
        # hidden: num_layers*2 by 1 by self.e_hidden_dim

        dim = self.e_embed_dim

        if self.pos_embed_dim is not None and poss is not None:
            output_ner, hidden_ner = self.nn_ner(embeds)
            output = torch.cat((output, output_ner), 2)
            hidden = torch.cat((hidden, hidden_ner), 2)
            dim += self.e_embed_dim
        if self.ner_embed_dim is not None and ners is not None:
            output_pos, hidden_pos = self.nn_pos(embeds)
            output = torch.cat((output, output_pos), 2)
            hidden = torch.cat((hidden, hidden_pos), 2)
            dim += self.e_embed_dim
        # output: seqlen by 1 by self.e_embed_dim * 3
        # hidden: num_layers*2 by 1 by self.e_hidden_dim * 3

        # concat to the same length
        # encoder_output: self.max_length by self.e_embed_dim * 3
        if length < self.max_length:
            encoder_output = torch.cat((torch.zeros(
                (self.max_length - length, dim), device=DEVICE), output.squeeze(1)), dim=0)
        else:
            encoder_output = output
        # output: max_length by self.e_embed_dim * 3
        # hidden: num_layers*2 by 1 by self.e_hidden_dim * 3
        returnn encoder_output, hidden

    def decode(self, hidden, targets:None):

        # 如果不是评估模式，则不可以使用teacher_forcing_ratio
        self.encoder_output = encoder_output

        ner_ts = targets
        if targets is not None
        and  random.random() < teacher_forcing_ratio:
            teacher = 

        if :
        for ix in range(length):

            decoder_output_tuple, decoder_output, decoder_hidden, decoder_attention = self._decode(
                decoder_input, decoder_hidden)
            if teacher:
                target_tuple = ()
                loss, decoder_input = self._decoding_teacher()
            else:
                loss, decoder_input, stop = self._decoding_normal()
                if stop:
                    break
        

    def _decoding_teacher(self, loss, decoder_output_tuple, target_tuple):
        """
        """
        word_output, ner_output, pos_output = decoder_output_tuple
        word_ts, ner_ts, pos_ts = target_tuple

        loss += criterion(word_output, word_tg)
        loss += criterion(ner_output, ner_tg)
        loss += criterion(pos_output, pos_tg)

        decoder_input = word_ts
        decoder_input = torch.cat((decoder_input, ner_ts), 1)
        decoder_input = torch.cat((decoder_input, pos_ts), 1)
        return loss, decoder_input

    def _decoding_normal(self, loss, decoder_output_tuple):
        """
        """
        word_output, ner_output, pos_output = decoder_output_tuple

        topv, topi = word_output.topk(1, dim=1)  # value, index
        word_input = topi.detach()
        decoder_input = word_input

        topv, topi = ner_output.topk(1, dim=1)  # value, index
        ner_input = topi.detach()
        decoder_input = torch.cat((decoder_input, ner_input), 1)

        topv, topi = pos_output.topk(1, dim=1)  # value, index
        pos_input = topi.detach()
        decoder_input = torch.cat((decoder_input, pos_input), 1)

        loss += self.criterion(word_output, word_tg[ix])
        loss += self.criterion(ner_output, ner_tg[ix])
        loss += self.criterion(pos_output, pos_tg[ix])

        stop = False
        if word_input.squeeze().item() == EOS_token:
            stop = True
        elif ner_input.squeeze().item() == EOS_token:
            stop = True
        elif pos_input.squeeze().item() == EOS_token:
            stop = True
        return loss, decoder_input, stop

    def _decode():

        for ix in range(length):

            # Get Final layer
            # hidden: num_layers*2 by 1 by self.e_hidden_dim * 3
            if self.bi:  # 这里要取出最后一层的正和逆两向，
                last_hidden = (hidden[num_layers - 1] * 0.5 + hidden[-1] * 0.5)
                last_hidden = last_hidden.unsqueeze(0)
            else:
                last_hidden = hidden[-1].unsqueeze(0)
            # last_hidden: 1 by 1 by self.e_hidden_dim * 3

            # Get embedding
            # embeds: 1 by 1 by self.e_embed_dim
            embeds = self._get_feature(words, ners, poss)

            # Get Attention
            # last_hidden: 1 by 1 by self.e_hidden_dim * 3
            # output: 1 by 1 by a_hidden_dim
            # attn_weights: 1 by self.max_length
            att_output, attn_weights = attention(embeds, last_hidden)

            # Use GRU
            # att_output: 1 by 1 by a_hidden_dim
            # hidden: num_layers*2 by 1 by self.e_hidden_dim * 3
            output, hidden = self.decode_gru(att_output, hidden)
            output = F.log_softmax(self.out(output[0]), dim=1)

            # Linear
            # output[0] 1 by hidden_size,
            output_word = F.log_softmax(self.out_word(output[0]), dim=1)
            output_final = output_word

            output_ner = None
            if self.ner_embed_dim:
                output_ner = F.log_softmax(self.out_ner(output[0]), dim=1)
                output_final = torch.cat((output_final, output_ner), 1)

            output_pos = None
            if self.pos_embed_dim:
                output_pos = F.log_softmax(self.out_pos(output[0]), dim=1)
                output_final = torch.cat((output_final, output_pos), 1)

            return (output_word, output_ner, output_pos), output_final, hidden, attn_weights

    def forward_loss(self):
        encoder_output, hidden = self.encode()

        = self.decode(encoder_output, hidden)

    def forward(self):
