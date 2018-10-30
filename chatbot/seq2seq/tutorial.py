# coding:utf8
# @Time    : 18-8-9 上午10:12
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH=20
SOS_token=0
EOS_token=1


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


teacher_forcing_ratio = 0.5
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, epochs, dataset, init_epochs, learning_rate=0.01):
    plot_losses = []

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(init_epochs, epochs+init_epochs):
        for i, (input_tensor, target_tensor) in enumerate(dataset.gen()):
            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            if loss:
                plot_losses.append(loss)
                if i % 1000==0:
                    print("epoch {}, step: {}, loss: {}".format(
                        epoch, i, loss
                    ))
            else:
                print(input_tensor, target_tensor)
        print("save model")
        torch.save(encoder.state_dict(), "epoch_{}_step_{}_encoder_loss_{}.pkl".format(epoch, i, loss))
        torch.save(decoder.state_dict(), "epoch_{}_step_{}_decoder_loss_{}.pkl".format(epoch, i, loss))


from chatbot.seq2seq.data import DataSet
from chatbot.utils.file import save_pickle

dataset = DataSet(MAX_LENGTH-1)
source_vocab = dataset.s_vocab
target_vocab = dataset.t_vocab
save_pickle(source_vocab, "source_vocab")
save_pickle(target_vocab, "target_vocab")
hidden_size = 256
encoder1 = EncoderRNN(dataset.s_vocab.size(), hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, dataset.t_vocab.size(), dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 5, dataset, init_epochs=19)


from chatbot.preprocessing.text import cut
from chatbot.seq2seq.data import tensorFromSentence




def evaluate(encoder, decoder, source_vocab, target_vocab, sentence, max_length=20):
    with torch.no_grad():
        input_tensor = tensorFromSentence(cut(sentence), source_vocab)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<eos>')
                break
            else:
                decoded_words.append(target_vocab.idx2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

print(evaluate(encoder1, attn_decoder1, source_vocab, target_vocab, "抱抱我")[0])
# print(evaluate(encoder1, attn_decoder1, source_vocab, "许兵到底是谁")[0])
print(evaluate(encoder1, attn_decoder1, source_vocab, target_vocab, "你是谁")[0])
print(evaluate(encoder1, attn_decoder1, source_vocab, target_vocab, "哈哈")[0])
print(evaluate(encoder1, attn_decoder1, source_vocab, target_vocab, "你真傻")[0])
print(evaluate(encoder1, attn_decoder1, source_vocab, target_vocab, "今天想吃牛肉火锅")[0])
print(evaluate(encoder1, attn_decoder1, source_vocab, target_vocab, "每天做梦一夜暴富")[0])
print(evaluate(encoder1, attn_decoder1, source_vocab, target_vocab, "我还喜欢她,怎么办")[0])
print(evaluate(encoder1, attn_decoder1, source_vocab, target_vocab, "干嘛阿。。")[0])
#
#
# def evaluateRandomly(encoder, decoder, n=10):
#     for i in range(n):
#         pair = random.choice(pairs)
#         print('>', pair[0])
#         print('=', pair[1])
#         output_words, attentions = evaluate(encoder, decoder, pair[0])
#         output_sentence = ' '.join(output_words)
#         print('<', output_sentence)
#         print('')
#


# def showAttention(input_sentence, output_words, attentions):
#     # Set up figure with colorbar
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(attentions.numpy(), cmap='bone')
#     fig.colorbar(cax)
#
#     # Set up axes
#     ax.set_xticklabels([''] + input_sentence.split(' ') +
#                        ['<EOS>'], rotation=90)
#     ax.set_yticklabels([''] + output_words)
#
#     # Show label at every tick
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#     plt.show()
#
#
# def evaluateAndShowAttention(input_sentence):
#     output_words, attentions = evaluate(
#         encoder1, attn_decoder1, input_sentence)
#     print('input =', input_sentence)
#     print('output =', ' '.join(output_words))
#     showAttention(input_sentence, output_words, attentions)
#
#
# evaluateAndShowAttention("elle a cinq ans de moins que moi .")
#
# evaluateAndShowAttention("elle est trop petit .")
#
# evaluateAndShowAttention("je ne crains pas de mourir .")
#
# evaluateAndShowAttention("c est un jeune directeur plein de talent .")
