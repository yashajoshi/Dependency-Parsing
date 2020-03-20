import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from general_utils import get_minibatches
from test_functions import get_UAS, compute_dependencies
from feature_extraction import load_datasets, DataConfig, Flags, punc_pos, pos_prefix
import argparse

from model import ParserModel

def eval(parser):
    # function for going through the steps of dependency parsing

    # load dataset
    load_existing_dump=True
    print('Loading data for testing')
    dataset = load_datasets(load_existing_dump)
    config = dataset.model_config

    idx2tran = {0:'LEFT-ARC',1:'RIGHT-ARC',2:'SHIFT'}
    sentence = dataset.test_data[0]
    print("Input sentence: ",[tok.word for tok in sentence.tokens])
    batch_sentences = [sentence]
    sentence.clear_prediction_dependencies()
    sentence.clear_children_info()
    enable_features=[]
    enable_features.append(0 if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1)
    enable_count = 1
    cnt = 0
    all_transitions = []
    all_dependencies = []
    while enable_count > 0:
        cnt+=1
        print('\n')
        print('----------------------------------------------')
        print('Step %d'%cnt)
        curr_sentences = []

        word_inputs_batch = []
        pos_inputs_batch = []
        dep_inputs_batch = []
        
        # If we still have parsing to do for this sentence
        if enable_features[0] == 1:
            curr_sentences.append(sentence)
            inputs = dataset.feature_extractor.extract_for_current_state( 
                sentence, dataset.word2idx, dataset.pos2idx, dataset.dep2idx)

            word_inputs_batch.append(inputs[0])
            pos_inputs_batch.append(inputs[1])
            dep_inputs_batch.append(inputs[2])
            
        print('Stack: ',[tok.word for tok in sentence.stack])
        print("Buff: ",[tok.word for tok in sentence.buff])
        print('Dependencies: ',[(action,pair[1].word,pair[0].word) for (action,pair) in \
                                zip(all_dependencies,sentence.predicted_dependencies)])
        print('Transitions: ',all_transitions)

        # get feature for each sentence
        # call predictions -> argmax
        # store dependency and left/right child
        # update state
        # repeat

        word_inputs_batch = torch.tensor(word_inputs_batch).to(device)
        pos_inputs_batch = torch.tensor(pos_inputs_batch).to(device)
        dep_inputs_batch = torch.tensor(dep_inputs_batch).to(device)

        # These are the raw outputs, which represent the activations for
        # prediction over valid transitions
        predictions = parser(word_inputs_batch,
                        pos_inputs_batch, dep_inputs_batch)
        legal_labels = np.asarray([sentence.get_legal_labels() for sentence in curr_sentences],
                                  dtype=np.float32)
        legal_transitions = np.argmax(predictions.cpu().detach().numpy() + 1000 * legal_labels, axis=1)
        pred = predictions.argmax().item()
        action = idx2tran[pred]
        print("Action: ",action)
        all_transitions.append(action)
        if not pred==2:
            all_dependencies.append(action)

        # update left/right children so can be used for next feature vector
        [sentence.update_child_dependencies(transition) for (sentence, transition) in
         zip(curr_sentences, legal_transitions) if transition != 2]

        # update state
        [sentence.update_state_by_transition(legal_transition, gold=False) for (sentence, legal_transition) in
         zip(curr_sentences, legal_transitions)]

        enable_features = [0 if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1 for sentence in
                           batch_sentences]
        enable_count = np.count_nonzero(enable_features)

    # print result at final state
    print('\n')
    print('----------------------------------------------')
    print('Step %d'%(cnt+1))
    print('Stack: ',[tok.word for tok in sentence.stack])
    print("Buff: ",[tok.word for tok in sentence.buff])
    print('Dependencies: ',[(action,pair[1].word,pair[0].word) for (action,pair) in \
                            zip(all_dependencies,sentence.predicted_dependencies)])
    print('Transitions: ',all_transitions)

    # reset sentence at end
    sentence.reset_to_initial_state()    

if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--load_model_file", help="Load the specified "
                           + "saved model for testing",
                           type=str, default=None)

    args = argparser.parse_args()
    load_file = args.load_model_file

    if load_file is None:
        # Back off to see if we can keep going
        load_file = 'saved_weights/parser-epoch-1.mdl'
    parser = torch.load(load_file)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    parser.to(device)
    eval(parser)





