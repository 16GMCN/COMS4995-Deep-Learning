import math
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from utils import CocoImageFolder, coco_eval, to_var
from data_loader import get_loader 
from adaptive import Encoder2Decoder
from build_vocab import Vocabulary
from torch.autograd import Variable 
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence

    
def main(args):
    
    # To reproduce training results
    torch.manual_seed( args.seed )
    if torch.cuda.is_available():
        torch.cuda.manual_seed( args.seed )
        
    # Create model directory
    if not os.path.exists( args.model_path ):
        os.makedirs(args.model_path)
    
    # Image Preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([ 
        transforms.RandomCrop( args.crop_size ),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(( 0.485, 0.456, 0.406 ), 
                             ( 0.229, 0.224, 0.225 ))])
    
    # Load vocabulary wrapper.
    with open( args.vocab_path, 'rb') as f:
        vocab = pickle.load( f )
    
    # Build training data loader
    data_loader = get_loader( args.image_dir, args.caption_path, vocab, 
                              transform, args.batch_size,
                              shuffle=True, num_workers=args.num_workers ) 

    # Load pretrained model or build from scratch
    adaptive = Encoder2Decoder( args.embed_size, len(vocab), args.hidden_size )
    
    if args.pretrained:
        
        adaptive.load_state_dict( torch.load( args.pretrained ) )
        # Get starting epoch #, note that model is named as '...your path to model/algoname-epoch#.pkl'
        # A little messy here.
        start_epoch = int( args.pretrained.split('/')[-1].split('-')[1].split('.')[0] ) + 1
        
    else:
        start_epoch = 1
    
    # Constructing CNN parameters for optimization, only fine-tuning higher layers
    cnn_subs = list( adaptive.encoder.resnet_conv.children() )[ args.fine_tune_start_layer: ]
    cnn_params = [ list( sub_module.parameters() ) for sub_module in cnn_subs ]
    cnn_params = [ item for sublist in cnn_params for item in sublist ]
    
    cnn_optimizer = torch.optim.Adam( cnn_params, lr=args.learning_rate_cnn, 
                                      betas=( args.alpha, args.beta ) )
    
    # Other parameter optimization
    params = list( adaptive.encoder.affine_a.parameters() ) + list( adaptive.encoder.affine_b.parameters() ) \
                + list( adaptive.decoder.parameters() )

    # Will decay later    
    learning_rate = args.learning_rate
    
    # Language Modeling Loss, Optimizers
    LMcriterion = nn.CrossEntropyLoss()
    
    # Change to GPU mode if available
    if torch.cuda.is_available():
        adaptive.cuda()
        LMcriterion.cuda()
    
    # Train the Models
    total_step = len( data_loader )
    
    bleu1_scores = []
    best_bleu1 = 0.0
    best_epoch = 0
    
    # Start Training 
    for epoch in range( start_epoch, args.num_epochs + 1 ):
        
        optimizer = torch.optim.Adam( params, lr=learning_rate )
        
        # Language Modeling Training
        print '------------------Training for Epoch %d----------------'%(epoch)
        for i, (images, captions, lengths, _ ) in enumerate( data_loader ):
            
            if(i > 100):
                break

            # Set mini-batch dataset
            images = to_var( images )
            captions = to_var( captions )
            lengths = [ cap_len - 1  for cap_len in lengths ]
            targets = pack_padded_sequence( captions[:,1:], lengths, batch_first=True )[0]

            # Forward, Backward and Optimize
            adaptive.train()
            adaptive.zero_grad()

            packed_scores = adaptive( images, captions, lengths )

            # Compute loss and backprop
            loss = LMcriterion( packed_scores[0], targets )
            loss.backward()
            
            # Gradient clipping for gradient exploding problem in LSTM
            for p in adaptive.decoder.LSTM.parameters():
                p.data.clamp_( -args.clip, args.clip )
            
            optimizer.step()
            
            # Start learning rate decay
            if epoch > args.lr_decay:
                
                frac = ( epoch - args.cnn_epoch ) / args.learning_rate_decay_every
                decay_factor = math.pow( 0.5, frac )

                # Decay the learning rate
                learning_rate = learning_rate * decay_factor
            
            # Start CNN fine-tuning
            if epoch > args.cnn_epoch:

                cnn_optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print 'Epoch [%d/%d], Step [%d/%d], CrossEntropy Loss: %.4f, Perplexity: %5.4f'%( epoch, 
                                                                                                 args.num_epochs, 
                                                                                                 i, total_step, 
                                                                                                 loss.data[0],
                                                                                                 np.exp( loss.data[0] ) )  
                
        # Save the Adaptive Attention model after each epoch
        torch.save( adaptive.state_dict(), 
                   os.path.join( args.model_path, 'adaptive-%d.pkl'%( epoch ) ) )
        
        # Evaluation on validation set        
        bleu1 = coco_eval( adaptive, args, epoch )
        bleu1_scores.append( bleu1 )        
        
        if bleu1 > best_bleu1:
            best_bleu1 = bleu1
            best_epoch = epoch
        print "BEST Bleu1 so far {}".format(best_bleu1)
       
#     # save bleus to file
#     resName = "./results/bleu1-%d.txt" % args.num_epochs
#     with open(resName, 'w') as file:
#         file.write(bleu1_scores)
#         file.write("\n")
#         file.write("Best: {} at epoch {}".format(best_bleu1, best_epoch))
        
#     print "{} is saved!".format(resName)
            
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument( '-f', default='self', help='To make it runnable in jupyter' )
    parser.add_argument( '--model_path', type=str, default='./models/',
                         help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/resized/train2014' ,
                        help='directory for resized training images')
    parser.add_argument('--val_dir', type=str, default='./data/resized/val2014',
                        help='directory for resized validation images' )
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--caption_val_path', type=str,
                        default='./data/annotations/captions_val2014.json',
                        help='path for validation annotation json file')
    parser.add_argument('--log_step', type=int, default=100,
                        help='step size for printing log info')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for model reproduction')
    
    # ---------------------------Hyper Parameter Setup------------------------------------
    
    # CNN fine-tuning
    parser.add_argument('--fine_tune_start_layer', type=int, default=6,
                        help='CNN fine-tuning layers from: [0-7]')
    parser.add_argument('--cnn_epoch', type=int, default=20,
                        help='start fine-tuning CNN after')
    
    # Optimizer Adam parameter
    parser.add_argument( '--alpha', type=float, default=0.8,
                         help='alpha in Adam' )
    parser.add_argument( '--beta', type=float, default=0.999,
                         help='beta in Adam' )
    parser.add_argument( '--learning_rate', type=float, default=4e-4,
                         help='learning rate for the whole model' )
    parser.add_argument( '--learning_rate_cnn', type=float, default=1e-4,
                         help='learning rate for fine-tuning CNN' )
    
    # LSTM hyper parameters
    parser.add_argument( '--embed_size', type=int, default=256,
                         help='dimension of word embedding vectors, also dimension of v_g' )
    parser.add_argument( '--hidden_size', type=int, default=512,
                         help='dimension of lstm hidden states' )
    
    # Training details
    parser.add_argument( '--pretrained', type=str, default='', help='start from checkpoint or scratch' )
    parser.add_argument( '--num_epochs', type=int, default=50 )
    parser.add_argument( '--batch_size', type=int, default=30 )
    
    # For eval_size > 30, it will cause cuda OOM error.
    parser.add_argument( '--eval_size', type=int, default=10 ) 
    parser.add_argument( '--num_workers', type=int, default=2 )
    parser.add_argument( '--clip', type=float, default=0.1 )
    parser.add_argument( '--lr_decay', type=int, default=20, help='epoch at which to start lr decay' )
    parser.add_argument( '--learning_rate_decay_every', type=int, default=50,
                         help='decay learning rate at every this number')
    
    args = parser.parse_args()
    
    print '------------------------Model and Training Details--------------------------'
    print(args)
    
    # Start training
    main( args )