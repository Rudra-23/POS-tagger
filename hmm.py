import pandas as pd
import json
import shutil
import os

'''
This will create a vocab.txt. 
inputs: filename: input_file (data/train)
threshold
output_file: file path for output in vocab.
'''
 
def create_vocab(filename, threshold, output_file):
    vocab = {}
    with open(filename, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.split('\t')
            if len(line) == 3:
                _, word, _ = line
                if not word in vocab.keys():
                    vocab[word] = 1
                else:
                    vocab[word] += 1
            

    df = pd.DataFrame(vocab.items(), columns = ['word', 'occurrence'])
    df['new_word'] = df.apply(lambda row: row['word'] if row['occurrence'] >= threshold else '<unk>', axis=1)

    unk_values = df[df['new_word'] == '<unk>'].shape[0]

    new_row_data = {'word': '<unk>', 'occurrence': unk_values, 'new_word': '<unk>'}
    new_row_df = pd.DataFrame(new_row_data, index=[0])

    df = df.drop(df[df['new_word'] == '<unk>'].index, axis = 0).sort_values(['occurrence'], ascending = False).reset_index(drop = True)
    df = pd.concat([new_row_df, df]).reset_index(drop=True)
    df['index'] = df.index
    
    df.to_csv(output_file, sep='\t', index = False, columns=['new_word', 'index', 'occurrence'], header = False)


'''
Returns vocab as dict.
input: file path of vocab.txt
'''
def get_vocab(filename):
    vocab = {}

    with open(filename, 'r') as f:
        lines = f.readlines()

        for line in lines:
            word, _ , occurence = line.split('\t')
            vocab[word] = int(occurence)

    return vocab

'''
 Returns all the states along with its count.
 input: training data (data/train)
'''
def get_states(filename):
    states = {}

    with open(filename, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip().split('\t')
            if len(line) == 3:
                idx, _, tag = line
                if not tag in states.keys():
                    states[tag] = 1
                else:
                    states[tag] += 1
        
    return states


'''
 Return sentences from training file. 
 input: training data (data/train)
'''
def get_sentences(filename):
    sentences = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        
        idx = 0
        while idx < len(lines):
            sentence = []
            while idx < len(lines) and lines[idx] != '\n':
                sentence.append(lines[idx])
                idx += 1
            sentences.append(sentence)
            idx += 1

    return sentences

'''
Return emission probablities using sentence, states, and vocab.
'''
def get_emissions(sentences, states, vocab):
    emissions_pairs = {}
    emissions = {}

    for sentence in sentences:
        for line in sentence:
            _, word, tag = line.strip().split('\t')

            if not word in vocab.keys():
                word = '<unk>'

            if (tag, word) in emissions_pairs.keys():
                emissions_pairs[(tag, word)] += 1
            else:
                emissions_pairs[(tag, word)] = 1

    for k, v in emissions_pairs.items():
        tag, word = k
        emissions[k] = v / states[tag]

    return emissions

'''
Return transition probablities using sentences and states.
'''
def get_transistions(sentences, states):
    transition_pairs = {}
    transitions = {}

    for sentence in sentences:
        for i in range(0, len(sentence)):
            tag_curr, tag_next = None, None
            if i > 0:
                curr, next = sentence[i-1], sentence[i]

                _, _, tag_curr = curr.strip().split('\t')
                _, _, tag_next = next.strip().split('\t')
            else:
                tag_curr = 'start'
                next = sentence[i]
                _, _, tag_next = next.strip().split('\t')

            if (tag_curr, tag_next) in transition_pairs.keys():
                transition_pairs[(tag_curr, tag_next)] += 1
            else:
                transition_pairs[(tag_curr, tag_next)] = 1

    for k, v in transition_pairs.items():
        tag_curr, tag_next = k
        transitions[k] = v / states[tag_curr]

    return transitions

'''
 Stores emissions and transitions in hmm.py
'''
def store_te(filename, emissions, transitions):
    transitions_str = {str(k): v for k, v in transitions.items()}
    emissions_str = {str(k): v for k, v in emissions.items()}

    with open(filename, 'w') as f:
        json.dump({'emissions': emissions_str, 'transitions': transitions_str}, f, indent=4)


'''
    Greedy Decoding
    Input:
        sentences, states, emissions, transitions, vocab, output filename, dev (dev = False, for test data).
'''
def greedy_decoding(sentences, states, emissions, transitions, vocab, filename, dev = True):
    f = open(filename, 'w')

    for sentence in sentences:
        prev_state = None

        for line in sentence:
            idx, word = None, None
            if dev:
                idx, word, _ = line.strip().split('\t')
            else:
                idx, word = line.strip().split('\t')

            original_word = word
            if not word in vocab:
                word = '<unk>' 

            max_state = None
            if prev_state == None:
                max_state = max(states, key=lambda s: transitions.get(('start', s), 0) * emissions.get((s, word), 0))
            else:
                max_state = max(states, key=lambda s: transitions.get((prev_state, s), 0) * emissions.get((s, word), 0))

            prev_state = max_state

            output = str(idx) + '\t' + str(original_word) + '\t' + str(max_state) + '\n'
            f.write(output)
        f.write('\n')
    f.close()


'''
    Viterbi Decoding
    Input:
        sentences, states, emissions, transitions, vocab, output filename, dev (dev = False, for test data).
'''        
def viterbi_decoding(sentences, states, emissions, transitions, vocab, filename, dev = True):
    f = open(filename, 'w')

    for sentence in sentences:
        dp = {}
        for i, line in enumerate(sentence):
            idx, word = None, None
            if dev:
                idx, word, _ = line.strip().split('\t')
            else:
                idx, word = line.strip().split('\t')

            original_word = word

            if not word in vocab:
                word = '<unk>' 

            if i == 0:
                for s in states.keys():
                    dp[(i, s)] = transitions.get(('start', s), 0) * emissions.get((s, word), 0)
            else:
                for s in states.keys():
                    mx = 0 
                    for s_hat in states.keys():
                        curr = dp.get((i-1, s_hat), 0) * transitions.get((s_hat, s), 0) * emissions.get((s, word), 0)
                        mx = max(mx, curr)
                    dp[(i, s)] = mx
            
            max_state = max(states, key=lambda s: dp.get((i, s)))
            output = str(idx) + '\t' + str(original_word) + '\t' + str(max_state) + '\n'
            f.write(output)
        f.write('\n')
    f.close()
    


'''
 Run your functions here
'''
if __name__ == '__main__':
    output_dir = './output/'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.mkdir(output_dir)

    create = True

    train_dir = './data/train'
    if create: 
        threshold = 2
        create_vocab(train_dir, threshold, output_dir + 'vocab.txt')

    vocab = get_vocab(output_dir + 'vocab.txt')
    print("Size of Vocabulary: ", len(vocab))

    states = get_states(train_dir)
    sentences = get_sentences('./data/train')

    '''
    'start' state will have count of number of sentences.
    '''
    states['start'] = len(sentences)

    emissions = get_emissions(sentences, states, vocab)
    print("No of parameters in emissions: ", len(emissions))

    transitions = get_transistions(sentences, states)
    print("No of parameters in transitions: ", len(transitions))

    store = True
    if store:
        store_te(output_dir + 'hmm.json', emissions, transitions)

    sentences = get_sentences('./data/dev')
    greedy_decoding(sentences, states, emissions, transitions, vocab,  output_dir + 'greedy_dev')
    viterbi_decoding(sentences, states, emissions, transitions, vocab,  output_dir + 'viterbi_dev')

    sentences = get_sentences('./data/test')
    greedy_decoding(sentences, states, emissions, transitions, vocab, output_dir + 'greedy.out', dev = False)
    viterbi_decoding(sentences, states, emissions, transitions, vocab, output_dir + 'viterbi.out', dev = False)



    

