import jsonlines
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

dev_file = 'Dataset/dev.jsonl'
test_file = 'Dataset/test.jsonl'
train_file = 'Dataset/train.jsonl'

#Constants

max_hypo_length, max_evid_length = 15, 15
batch_size, vector_size, hidden_size = 128, 50, 100
lstm_size = hidden_size
weight_decay = 0.0001
learning_rate = 1
input_p, output_p = 0.5, 0.5
training_iterations_count = 100000
layers = 2
display_step = 10

vector_dict = {}
count = 0

with open('Dataset/glove/glove.6B.50d.txt', "r", encoding='UTF-8') as glove:
    for line in glove:
        # if count<10:
        # print(line)
        name, vector = line.split(" ", 1)
        # print(name)
        # print(vector)
        vector_dict[name] = np.fromstring(vector, sep=" ")

def sentence2sequence(sentence):
    # print(sentence)
    tokens = sentence.lower()
    tokens = tokens.split(" ")
    rows = []
    words = []
    for token in tokens:
        i = len(token)
        while len(token)>0 and i>0:
            word = token[:i]
            if word in vector_dict:
                rows.append(vector_dict[word])
                words.append(word)
                token = token[i:]
                i = len(token)
            else:
                i = i-1
    return rows, words

def score_setup(row):
    # print(" Last row:", row)
    convert_dict = {
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }
    score = np.zeros((3,))
    tag =row["gold_label"]

    if tag in convert_dict: score[convert_dict[tag]] += 1
    else:
        score[1] += 1
    return score

def fit_to_size(matrix, shape):
    res = np.zeros(shape)
    slices = tuple([slice(0,min(dim,shape[e])) for e, dim in enumerate(matrix.shape)])
    res[slices] = matrix[slices]
    return res


def split_data_into_scores():
    # import csv
    with jsonlines.open(test_file,"r") as data:
        # train = csv.DictReader(data, delimiter='\t')
        evi_sentences = []
        hyp_sentences = []
        labels = []
        scores = []
        for row in data:
            # print(row)
            hyp_sentences.append(np.vstack(
                    sentence2sequence(row["sentence1"].lower())[0]))
            evi_sentences.append(np.vstack(
                    sentence2sequence(row["sentence2"].lower())[0]))
            labels.append(row["gold_label"])
            scores.append(score_setup(row))

        hyp_sentences = np.stack([fit_to_size(x, (max_hypo_length, vector_size))
                          for x in hyp_sentences])
        evi_sentences = np.stack([fit_to_size(x, (max_evid_length, vector_size))
                          for x in evi_sentences])

        return (hyp_sentences, evi_sentences), labels, np.array(scores)

data_feature_list, correct_values, correct_scores = split_data_into_scores()

# print(data_feature_list, correct_values, correct_scores)

l_h, l_e = max_hypo_length, max_evid_length
N, D, H = batch_size, vector_size, hidden_size
l_seq = l_h + l_e

def lstm_layer():
    return tf.nn.rnn_cell.LSTMCell(lstm_size)

lstm = tf.nn.rnn_cell.LSTMCell(lstm_size)
    # [lstm_layer() for _ in range(num_of_layers)])

lstm_drop = tf.contrib.rnn.DropoutWrapper(lstm, input_p, output_p)

hyp = tf.placeholder(tf.float32, [N, l_h, D], 'hypothesis')
evi = tf.placeholder(tf.float32, [N, l_e, D], 'evidence')
y = tf.placeholder(tf.float32, [N, 3], 'label')

lstm_back = tf.nn.rnn_cell.LSTMCell(lstm_size)

lstm_drop_back = tf.contrib.rnn.DropoutWrapper(lstm_back, input_p, output_p)

fc_initializer = tf.random_normal_initializer(stddev=0.1)

fc_init2 = tf.random_normal_initializer(stddev=0.1)

fc_weight = tf.get_variable('fc_weight', [2*hidden_size, 300],
                            initializer = fc_initializer)
fc_weight2 = tf.get_variable('fc_weight2', [300, 3],
                            initializer = fc_init2)

fc_bias = tf.get_variable('bias', [300])

fc_bias2 = tf.get_variable('bias1', [3])

tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                     tf.nn.l2_loss(fc_weight))

x = tf.concat([hyp, evi], 1)

x = tf.transpose(x, [1, 0, 2])

x = tf.reshape(x, [-1, vector_size])

x = tf.split(x, l_seq,)

rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm, lstm_back,
                                                            x, dtype=tf.float32)

classification_scores = tf.matmul(rnn_outputs[-1], fc_weight) + fc_bias

classification_sc2 = tf.matmul(classification_scores, fc_weight2) + fc_bias2

with tf.variable_scope('Accuracy'):
    predicts = tf.cast(tf.argmax(classification_sc2, 1), 'int32')
    y_label = tf.cast(tf.argmax(y, 1), 'int32')
    corrects = tf.equal(predicts, y_label)
    num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

with tf.variable_scope("loss"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits = classification_sc2, labels = y)
    loss = tf.reduce_mean(cross_entropy)
    total_loss = loss + weight_decay * tf.add_n(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

opt_op = optimizer.minimize(total_loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)


try:
    saver.restore(sess, "Dataset/trained3/model.ckpt")

except:

    training_iterations = range(0,training_iterations_count,batch_size)
    training_iterations = tqdm(training_iterations)

    for i in training_iterations:

        batch = np.random.randint(data_feature_list[0].shape[0], size=batch_size)

        hyps, evis, ys = (data_feature_list[0][batch,:],
                          data_feature_list[1][batch,:],
                          correct_scores[batch])

        sess.run([opt_op], feed_dict={hyp: hyps, evi: evis, y: ys})
        if (i/batch_size) % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={hyp: hyps, evi: evis, y: ys})
            # Calculate batch loss
            tmp_loss = sess.run(loss, feed_dict={hyp: hyps, evi: evis, y: ys})
            # Display results
            print("Iter " + str(i/batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(tmp_loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

evidences = ["well see that isn't too bad  a couple hours"]

hypotheses = ["I don't like waiting, not even for 10 minutes"]

sentence1 = [fit_to_size(np.vstack(sentence2sequence(evidence)[0]),
                         (15, 50)) for evidence in evidences]

sentence2 = [fit_to_size(np.vstack(sentence2sequence(hypothesis)[0]),
                         (15,50)) for hypothesis in hypotheses]

prediction = sess.run(classification_sc2, feed_dict={hyp: (sentence1 * batch_size),
                                                    evi: (sentence2 * batch_size),
                                                    y: [[0,0,0]]*batch_size})
print(["Entailment", "Neutral", "Contradict"][np.argmax(prediction[0])])
print(prediction[0])

save_path = saver.save(sess,"Dataset/trained2/model.ckpt")
sess.close()
