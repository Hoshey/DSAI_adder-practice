####################################
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range
####################################
class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


TRAINING_SIZE = 50000
DIGITS = 3

REVERSE = True

MAXLEN = DIGITS + 1 + DIGITS
chars = '0123456789+ '

RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
####################################
class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)

ctable = CharacterTable(chars)
####################################
questions = []
expected = []
seen = set()

print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789'))
                            for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    ans += ' ' * (DIGITS + 1 - len(ans))

    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))
print(questions[:5], expected[:5])
####################################
print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(expected), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)
print('Done...')
####################################
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# train_test_split
train_x = x[:30000]
train_y = y[:30000]
test_x = x[30000:]
test_y = y[30000:]

split_at = len(train_x) - len(train_x) // 10
(x_train, x_val) = train_x[:split_at], train_x[split_at:]
(y_train, y_val) = train_y[:split_at], train_y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Testing Data:')
print(test_x.shape)
print(test_y.shape)
####################################
##### Build your own model here ############
print('Build model...')
model = Sequential()

# "Encode" the input sequence using an RNN, input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))

# the decoder RNN's input
model.add(layers.RepeatVector(DIGITS + 1))

# The decoder RNN  with a single layer.
for _ in range(LAYERS):
    # all the outputs in the form of (num_samples, timesteps, output_dim). 
    # the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))


model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
####################################
for iteration in range(50):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=3,
              validation_data=(x_val, y_val))
    # visualizing validation set prediction errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)

####################################
*print("MSG : Prediction")
## Try to test and evaluate your model ##############
preds = model.predict_classes(test_x, verbose=0)

i = 0
count = 0
while i < len(preds) :
    q = ctable.decode(test_x[i])
    correct = ctable.decode(test_y[i])
    guess = ctable.decode(preds[i], calc_argmax=False)

    print('Q', q, end=' ')
    print('T', correct, end=' ')
    if correct == guess:
        print(colors.ok + '☑' + colors.close, end=' ')
        count += 1
    else:
        print(colors.fail + '☒' + colors.close, end=' ')
    print(guess)
    i += 1
    
testing_acc = (count/len(test_y))*100
print('-'*60)
print('-'*30, colors.ok + str(testing_acc) + "%" + colors.close, '-'*30)
print('-'*60)
#####################################################



