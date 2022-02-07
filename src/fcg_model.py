from typing import Dict, Optional

from fairseq.models import FairseqEncoderDecoderModel, FairseqEncoder, FairseqDecoder, register_model, register_model_architecture
from fairseq.models.lstm import LSTMModel, LSTMEncoder, LSTMDecoder
import torch
import torch.nn as nn
from torch import Tensor
from fairseq import utils


class SourceEncoder(LSTMEncoder):
    def __init__(
            self,
            dictionary,
            embed_dim=512,
            err_embed_dim=100,
            hidden_size=512,
            num_layers=1,
            dropout_in=0.1,
            dropout_out=0.1,
            bidirectional=False,
            left_pad=True,
            pretrained_embed=None,
            padding_idx=None,
            max_source_positions=1e5,
    ):
        super().__init__(dictionary, embed_dim, hidden_size, num_layers, dropout_in, dropout_out, bidirectional,
                         left_pad, pretrained_embed, padding_idx, max_source_positions)

        lstm = nn.LSTM(
            embed_dim + err_embed_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out_module.p if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        for name, param in lstm.named_parameters():
            if "weight" in name or "bias" in name:
                param.data.uniform_(-0.1, 0.1)
        self.lstm = lstm
        self.embed_err = nn.Embedding(2, err_embed_dim, padding_idx=dictionary.pad())

    def forward(
            self,
            src_tokens,
            src_lengths,
            src_errors,
            enforce_sorted=True,
    ):
        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                torch.zeros_like(src_tokens).fill_(self.padding_idx),
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = self.dropout_in_module(x)

        errors = torch.zeros_like(src_tokens)
        for err, o, l in zip(errors, src_errors, src_lengths):
            err[o[0]:o[1]] = 1
            err[l:] = self.dictionary.pad()
        e_errors = self.embed_err(errors)
        e_errors = self.dropout_in_module(e_errors)

        x = torch.cat((x, e_errors), dim=2)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, src_lengths.cpu(), enforce_sorted=enforce_sorted
        )

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outs, padding_value=self.padding_idx * 1.0
        )
        x = self.dropout_out_module(x)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:
            final_hiddens = self.combine_bidir(final_hiddens, bsz)
            final_cells = self.combine_bidir(final_cells, bsz)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        _x = x.transpose(0, 1)
        final_hiddens = torch.stack([torch.mean(y[o[0]:o[1]], dim=0) for o, y in zip(src_errors, _x)]).unsqueeze(0)

        return tuple(
            (
                x,  # seq_len x batch x hidden
                final_hiddens,  # num_layers x batch x num_directions*hidden
                final_cells,  # num_layers x batch x num_directions*hidden
                encoder_padding_mask,  # seq_len x batch
                src_tokens
            )
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),
                encoder_out[1].index_select(1, new_order),
                encoder_out[2].index_select(1, new_order),
                encoder_out[3].index_select(1, new_order),
                encoder_out[4].index_select(0, new_order),
            )
        )


class PointerGeneratorDecoder(LSTMDecoder):
    def __init__(
        self,
        dictionary,
        embed_dim=512,
        hidden_size=512,
        out_embed_dim=512,
        num_layers=1,
        dropout_in=0.1,
        dropout_out=0.1,
        attention=True,
        encoder_output_units=512,
        pretrained_embed=None,
        share_input_output_embed=False,
        adaptive_softmax_cutoff=None,
        max_target_positions=1e5,
        residuals=False,
    ):
        super().__init__(dictionary, embed_dim, hidden_size, out_embed_dim, num_layers, dropout_in, dropout_out,
                         attention, encoder_output_units, pretrained_embed, share_input_output_embed,
                         adaptive_softmax_cutoff, max_target_positions, residuals)

        self.vocab_size = len(dictionary)
        # self.W_s = nn.Linear(hidden_size, vocab_size)
        # self.W_c = nn.Linear(hidden_size + encoder_output_units, hidden_size)
        # self.W_att_h = nn.Linear(encoder_output_units, hidden_size)
        # self.W_att_s = nn.Linear(hidden_size, hidden_size)
        # self.v_att = nn.Linear(hidden_size, 1)

        # self.w_h = nn.Linear(out_embed_dim, 1)
        # self.w_s = nn.Linear(hidden_size, 1)
        # self.w_x = nn.Linear(embed_dim, 1)

        self.w_h = nn.Linear(out_embed_dim, hidden_size)
        self.w_s = nn.Linear(hidden_size, hidden_size)
        self.w_x = nn.Linear(embed_dim, hidden_size)
        self.w_p_gen = nn.Linear(hidden_size, 1)

        # self.attention = AttentionLayer(
        #     hidden_size, encoder_output_units, hidden_size, bias=False
        # )

    def forward(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        src_lengths=None,
    ):
        final_hiddens, attn_scores, lstm_hiddens, e = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        # p_gen = torch.sigmoid(self.w_h(final_hiddens) + self.w_s(lstm_hiddens) + self.w_x(e))
        p_gen = torch.sigmoid(self.w_p_gen(torch.tanh(self.w_h(final_hiddens) + self.w_s(lstm_hiddens) + self.w_x(e))))

        src_tokens = encoder_out[-1]
        onehot = nn.functional.one_hot(src_tokens, num_classes=self.vocab_size).float()
        source_dist = torch.matmul(attn_scores, onehot)

        vocab_dist = self.output_layer(final_hiddens)

        final_dist = p_gen * vocab_dist + (1 - p_gen) * source_dist

        return final_dist, attn_scores

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
    ):
        """
        Similar to *forward* but only return features.
        """
        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs = encoder_out[0]
            encoder_hiddens = encoder_out[1]
            encoder_cells = encoder_out[2]
            encoder_padding_mask = encoder_out[3]
        else:
            encoder_outs = torch.empty(0)
            encoder_hiddens = torch.empty(0)
            encoder_cells = torch.empty(0)
            encoder_padding_mask = torch.empty(0)
        srclen = encoder_outs.size(0)

        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        e = self.embed_tokens(prev_output_tokens)
        e = self.dropout_in_module(e)

        # B x T x C -> T x B x C
        x = e.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells, input_feed = self.get_cached_state(
                incremental_state
            )
        elif encoder_out is not None:
            # setup recurrent cells
            prev_hiddens = [encoder_hiddens[i] for i in range(self.num_layers)]
            prev_cells = [encoder_cells[i] for i in range(self.num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(y) for y in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(y) for y in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)
        else:
            # setup zero cells, since there is no encoder
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]
            input_feed = None

        assert (
            srclen > 0 or self.attention is None
        ), "attention is not supported if there are no encoder outputs"
        attn_scores = (
            x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        )
        outs = []
        hiddens = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input = x[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hidden)
                if self.residuals:
                    input = input + prev_hiddens[i]

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                assert attn_scores is not None
                out, attn_scores[:, j, :] = self.attention(
                    hidden, encoder_outs, encoder_padding_mask
                )
            else:
                out = hidden
            out = self.dropout_out_module(out)

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final output
            outs.append(out)
            hiddens.append(hidden)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        prev_cells_tensor = torch.stack(prev_cells)
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": prev_hiddens_tensor,
                "prev_cells": prev_cells_tensor,
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)
        hiddens = torch.cat(hiddens, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        hiddens = hiddens.transpose(1, 0)

        if hasattr(self, "additional_fc") and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = self.dropout_out_module(x)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if self.attention is not None:
            assert attn_scores is not None
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None
        return x, attn_scores, hiddens, e

    def reorder_incremental_state(
            self,
            incremental_state,
            new_order,
    ):
        if incremental_state is None or len(incremental_state) == 0:
            return
        prev_hiddens, prev_cells, input_feed = self.get_cached_state(incremental_state)
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]
        prev_cells = [p.index_select(0, new_order) for p in prev_cells]
        if input_feed is not None:
            input_feed = input_feed.index_select(0, new_order)
        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": torch.stack(prev_hiddens),
                "prev_cells": torch.stack(prev_cells),
                "input_feed": input_feed,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cached_state_new),
        return

def Linear(in_features, out_features, bias=True, dropout=0.0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

# import torch.nn.functional as F
#
# class AttentionLayer(nn.Module):
#     def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
#         super().__init__()
#
#         self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
#         self.output_proj = Linear(
#             input_embed_dim + source_embed_dim, output_embed_dim, bias=bias
#         )
#
#     def forward(self, input, source_hids, encoder_padding_mask):
#         # input: bsz x input_embed_dim
#         # source_hids: srclen x bsz x source_embed_dim
#
#         # x: bsz x source_embed_dim
#         x = self.input_proj(input)
#
#         # compute attention
#         attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
#
#         # don't attend over padding
#         if encoder_padding_mask is not None:
#             attn_scores = (
#                 attn_scores.float()
#                 .masked_fill_(encoder_padding_mask, float("-inf"))
#                 .type_as(attn_scores)
#             )  # FP16 support: cast to float and back
#
#         attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz
#
#         # sum weighted sources
#         x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)
#
#         x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
#         return x, attn_scores

@register_model('fcg1')
class PointerGenerator(LSTMModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--err_embed_dim', type=int, default=100,
            # help='dimensionality of the hidden state',
        )
        parser.add_argument(
            '--encoder_embed_path', type=str,
            # help='dimensionality of the hidden state',
        )

    @classmethod
    def build_model(cls, args, task):

        # Return the wrapped version of the module
        base_architecture(args)

        if args.encoder_layers != args.decoder_layers:
            raise ValueError("--encoder-layers must match --decoder-layers")

        max_source_positions = getattr(
            args, "max_source_positions", 1e5
        )
        max_target_positions = getattr(
            args, "max_target_positions", 1e5
        )

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = nn.Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim
            )
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = nn.Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError("--share-all-embeddings requires a joint dictionary")
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embed not compatible with --decoder-embed-path"
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to "
                    "match --decoder-embed-dim"
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim,
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
            args.decoder_embed_dim != args.decoder_out_embed_dim
        ):
            raise ValueError(
                "--share-decoder-input-output-embeddings requires "
                "--decoder-embed-dim to match --decoder-out-embed-dim"
            )

        if args.encoder_freeze_embed:
            pretrained_encoder_embed.weight.requires_grad = False
        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False


        encoder = SourceEncoder(
            dictionary=task.source_dictionary,
            err_embed_dim=args.err_embed_dim,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            max_source_positions=max_source_positions,
        )
        decoder = PointerGeneratorDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == "adaptive_loss"
                else None
            ),
            max_target_positions=max_target_positions,
            residuals=False,
        )
        return cls(encoder, decoder)

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            src_errors,
            incremental_state=None,
    ):
        encoder_out = self.encoder(src_tokens, src_lengths, src_errors)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
        )
        return decoder_out


@register_model_architecture('fcg1', 'fcg')
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 200)
    args.err_embed_dim = getattr(args, "err_embed_dim", 100)
    args.decoder_embed_dim = getattr(args, "encoder_hidden_size", 300)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_freeze_embed = getattr(args, "encoder_freeze_embed", False)
    args.encoder_hidden_size = getattr(
        args, "encoder_hidden_size", args.encoder_embed_dim
    )
    args.encoder_layers = getattr(args, "encoder_layers", 1)
    args.encoder_bidirectional = getattr(args, "encoder_bidirectional", True)
    args.encoder_dropout_in = getattr(args, "encoder_dropout_in", args.dropout)
    args.encoder_dropout_out = getattr(args, "encoder_dropout_out", args.dropout)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 200)
    args.decoder_embed_dim = getattr(args, "decoder_hidden_size", 300)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.decoder_hidden_size = getattr(
        args, "decoder_hidden_size", args.decoder_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 200)
    args.decoder_attention = getattr(args, "decoder_attention", "1")
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "10000,50000,200000"
    )


