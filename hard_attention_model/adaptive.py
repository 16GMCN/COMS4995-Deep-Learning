import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import copy


#========================================Knowing When to Look========================================
class AttentiveCNN( nn.Module ):
    def __init__( self, embed_size, hidden_size ):
        super( AttentiveCNN, self ).__init__()
        
        # ResNet-152 backend
        resnet = models.resnet152( pretrained=True )
        modules = list( resnet.children() )[ :-2 ] # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential( *modules ) # last conv feature
        
        self.resnet_conv = resnet_conv
        self.avgpool = nn.AvgPool2d( 7 )
        self.affine_a = nn.Linear( 2048, hidden_size ) # v_i = W_a * A
        self.affine_b = nn.Linear( 2048, embed_size )  # v_g = W_b * a^g
        
        # Dropout before affine transformation
        self.dropout = nn.Dropout( 0.5 )
        
        self.init_weights()
        
    def init_weights( self ):
        """Initialize the weights."""
        init.kaiming_uniform( self.affine_a.weight, mode='fan_in' )
        init.kaiming_uniform( self.affine_b.weight, mode='fan_in' )
        self.affine_a.bias.data.fill_( 0 )
        self.affine_b.bias.data.fill_( 0 )
        
        
    def forward( self, images ):
        '''
        Input: images
        Output: V=[v_1, ..., v_n], v_g
        '''
        
        # Last conv layer feature map
        A = self.resnet_conv( images )
        
        # a^g, average pooling feature map
        a_g = self.avgpool( A )
        a_g = a_g.view( a_g.size(0), -1 )
        
        # V = [ v_1, v_2, ..., v_49 ]
        V = A.view( A.size( 0 ), A.size( 1 ), -1 ).transpose( 1,2 )
        #V = F.relu( self.affine_a( self.dropout( V ) ) )
        
        v_g = F.relu( self.affine_b( self.dropout( a_g ) ) )
        
        return V, v_g

# Attention Block for C_hat calculation
class Atten( nn.Module ):
    def __init__( self, hidden_size ):
        super( Atten, self ).__init__()

        self.affine_v = nn.Linear( 2048, 49, bias=False ) # W_v
        self.affine_g = nn.Linear( hidden_size, 49, bias=False ) # W_g
        self.affine_s = nn.Linear( hidden_size, 49, bias=False ) # W_s
        self.affine_h = nn.Linear( 49, 1, bias=False ) # w_h
        self.affine_feature = nn.Linear( 1, 512, bias=False ) # w_h
        self.affine_cxt = nn.Linear( 512, 1, bias=False ) # w_h
        self.affine_g2 = nn.Linear( hidden_size, 512, bias=False ) # w_h
        self.affine_spatial = nn.Linear( 2048, 512, bias=False ) # w_h
        self.affine_channel = nn.Linear( 49, 512, bias=False ) # w_h
        self.gate_v_s = nn.Linear( hidden_size, 512, bias=False ) # w_h
        self.gate_v_c = nn.Linear( hidden_size, 512, bias=False ) # w_h
        self.gate_h_s = nn.Linear( hidden_size, 512, bias=False ) # w_h
        self.gate_h_c = nn.Linear( hidden_size, 512, bias=False ) # w_h
        self.dropout = nn.Dropout( 0.5 )
        self.init_weights()
        
    def init_weights( self ):
        """Initialize the weights."""
        init.xavier_uniform( self.affine_v.weight )
        init.xavier_uniform( self.affine_g.weight )
        init.xavier_uniform( self.affine_h.weight )
        init.xavier_uniform( self.affine_s.weight )
        
    def forward( self, V, h_t, s_t ):
        '''
        Input: V=[v_1, v_2, ... v_k], h_t, s_t from LSTM
        Output: c_hat_t, attention feature map
        '''
        
        featuremean = torch.mean(V,1).view(-1,V.size(2),1)
        feature_vec = self.affine_feature(self.dropout(featuremean)).unsqueeze(1)
        cxt_ = F.tanh(feature_vec+ self.affine_g2( self.dropout( h_t ) ).unsqueeze( 2 ))
        cxt = self.affine_cxt(self.dropout(cxt_)).squeeze(3)
        alpha0 =  F.softmax( cxt.view( -1, cxt.size( 2 ) ) ).view( cxt.size( 0 ), cxt.size( 1 ), -1 )
        weightedfeature = alpha0.unsqueeze(2) * V.unsqueeze(1)
        
        
        # W_v * V + W_g * h_t * 1^T
        content_v = F.tanh(self.affine_v( self.dropout( weightedfeature ) ) \
                    + self.affine_g( self.dropout( h_t ) ).unsqueeze( 2 ))
        
        # z_t = W_h * tanh( content_v )
        z_t = self.affine_h( self.dropout( F.tanh( content_v ) ) ).squeeze( 3 )
        # get mean at dim=2
        #z_t_ = torch.mean(z_t, dim=2, keepdim=True)
        alpha_t = F.softmax( z_t.view( -1, z_t.size( 2 ) ) ).view( z_t.size( 0 ), z_t.size( 1 ), -1 )
        
        # Construct c_t: B x seq x hidden_size
        c_t_spatial = torch.bmm( alpha_t, V ).squeeze( 2 )
        c_t_channel = alpha0.unsqueeze(2) * V.unsqueeze(1)
        c_t_channel = torch.mean(c_t_channel,3)
        spatial_info = self.affine_spatial(self.dropout(c_t_spatial))
        channel_info = self.affine_channel(self.dropout(c_t_channel))
        s_gate = F.sigmoid(self.gate_v_s(self.dropout(s_t))+self.gate_h_s(self.dropout(h_t)))
        c_gate = F.sigmoid(self.gate_v_c(self.dropout(s_t))+self.gate_h_c(self.dropout(h_t)))
        
        
        
        # W_s * s_t + W_g * h_t
        content_s = self.affine_s( self.dropout( s_t ) ) + self.affine_g( self.dropout( h_t ) )
        # w_t * tanh( content_s )
        z_t_extended = self.affine_h( self.dropout( F.tanh( content_s ) ) )
        
        # Attention score between sentinel and image content
        extended = torch.cat( ( z_t, z_t_extended ), dim=2 )
        alpha_hat_t = F.softmax( extended.view( -1, extended.size( 2 ) ) ).view( extended.size( 0 ), extended.size( 1 ), -1 )
        beta_t = alpha_hat_t[ :, :, -1 ]
        
        # c_hat_t = beta * s_t + ( 1 - beta ) * c_t
        beta_t = beta_t.unsqueeze( 2 )
        #c_hat_t = beta_t * s_t + ( 1 - beta_t ) * self.affine_spatial(self.dropout(c_t_spatial))
        c_hat_t = s_gate*spatial_info + c_gate*channel_info
        # c_hat_t = s_t + c_t + h_t

        return c_hat_t, alpha_t, beta_t

# Sentinel BLock    
class Sentinel( nn.Module ):
    def __init__( self, input_size, hidden_size ):
        super( Sentinel, self ).__init__()

        self.affine_x = nn.Linear( input_size, hidden_size, bias=False )
        self.affine_h = nn.Linear( hidden_size, hidden_size, bias=False )
        
        # Dropout applied before affine transformation
        self.dropout = nn.Dropout( 0.5 )
        
        self.init_weights()
        
    def init_weights( self ):
        init.xavier_uniform( self.affine_x.weight )
        init.xavier_uniform( self.affine_h.weight )
        
    def forward( self, x_t, h_t_1, cell_t ):
        
        # g_t = sigmoid( W_x * x_t + W_h * h_(t-1) )        
        gate_t = self.affine_x( self.dropout( x_t ) ) + self.affine_h( self.dropout( h_t_1 ) )
        gate_t = F.sigmoid( gate_t )
        
        # Sentinel embedding
        s_t =  gate_t * F.tanh( cell_t )
        
        return s_t

# Adaptive Attention Block: C_t, Spatial Attention Weights, Sentinel embedding    
class AdaptiveBlock( nn.Module ):
    
    def __init__( self, embed_size, hidden_size, vocab_size ):
        super( AdaptiveBlock, self ).__init__()

        # Sentinel block
        self.sentinel = Sentinel( embed_size * 2, hidden_size )
        
        # Image Spatial Attention Block
        self.atten = Atten( hidden_size )
        
        # Final Caption generator
        self.mlp = nn.Linear( hidden_size, vocab_size )
        
        # Dropout layer inside Affine Transformation
        self.dropout = nn.Dropout( 0.5 )
        
        self.hidden_size = hidden_size
        self.init_weights()
        
    def init_weights( self ):
        '''
        Initialize final classifier weights
        '''
        init.kaiming_normal( self.mlp.weight, mode='fan_in' )
        self.mlp.bias.data.fill_( 0 )
        
        
    def forward( self, x, hiddens, cells, V ):
        
        # hidden for sentinel should be h0-ht-1
        h0 = self.init_hidden( x.size(0) )[0].transpose( 0,1 )
        
        # h_(t-1): B x seq x hidden_size ( 0 - t-1 )
        if hiddens.size( 1 ) > 1:
            hiddens_t_1 = torch.cat( ( h0, hiddens[ :, :-1, : ] ), dim=1 )
        else:
            hiddens_t_1 = h0

        # Get Sentinel embedding, it's calculated blockly    
        sentinel = self.sentinel( x, hiddens_t_1, cells )
        
        # Get C_t, Spatial attention, sentinel score
        c_hat, atten_weights, beta = self.atten( V, hiddens, sentinel )
        
        # Final score along vocabulary
        scores = self.mlp( self.dropout( c_hat + hiddens ) )
#         scores = self.mlp( self.dropout( torch.cat(c_hat, hiddens, 2) ) )
        
        return scores, atten_weights, beta
    
    def init_hidden( self, bsz ):
        '''
        Hidden_0 & Cell_0 initialization
        '''
        weight = next( self.parameters() ).data
        
        if torch.cuda.is_available():
            return ( Variable( weight.new( 1 , bsz, self.hidden_size ).zero_().cuda() ),
                    Variable( weight.new( 1,  bsz, self.hidden_size ).zero_().cuda() ) ) 
        else: 
            return ( Variable( weight.new( 1 , bsz, self.hidden_size ).zero_() ),
                    Variable( weight.new( 1,  bsz, self.hidden_size ).zero_() ) ) 
    

# Caption Decoder
class Decoder( nn.Module ):
    def __init__( self, embed_size, vocab_size, hidden_size ):
        super( Decoder, self ).__init__()

        # word embedding
        self.embed = nn.Embedding( vocab_size, embed_size )
        
        # LSTM decoder: input = [ w_t; v_g ] => 2 x word_embed_size;
        self.LSTM = nn.LSTM( embed_size * 2, hidden_size, 1, batch_first=True )
        
        # Save hidden_size for hidden and cell variable 
        self.hidden_size = hidden_size
        
        # Adaptive Attention Block: Sentinel + C_hat + Final scores for caption sampling
        self.adaptive = AdaptiveBlock( embed_size, hidden_size, vocab_size )
        
    def forward( self, V, v_g , captions, states=None ):
        
        # Word Embedding
        embeddings = self.embed( captions )
        
        # x_t = [w_t;v_g]
        x = torch.cat( ( embeddings, v_g.unsqueeze( 1 ).expand_as( embeddings ) ), dim=2 )
        
        # Hiddens: Batch x seq_len x hidden_size
        # Cells: seq_len x Batch x hidden_size, default setup by Pytorch
        if torch.cuda.is_available():
            hiddens = Variable( torch.zeros( x.size(0), x.size(1), self.hidden_size ).cuda() )
            cells = Variable( torch.zeros( x.size(1), x.size(0), self.hidden_size ).cuda() )
        else:
            hiddens = Variable( torch.zeros( x.size(0), x.size(1), self.hidden_size ) )
            cells = Variable( torch.zeros( x.size(1), x.size(0), self.hidden_size ) )            
        
        # Recurrent Block
        # Retrieve hidden & cell for Sentinel simulation
        for time_step in range( x.size( 1 ) ):
            
            # Feed in x_t one at a time
            x_t = x[ :, time_step, : ]
            x_t = x_t.unsqueeze( 1 )
            
            h_t, states = self.LSTM( x_t, states )
            #h_t, states = self.LSTM( torch.cat([x_t,v_g.view(-1,1,v_g.size(1))],2), states )
            
#             print("STATE SIZE:", np.shape(states))
            
            # Save hidden and cell
            hiddens[ :, time_step, : ] = h_t  # Batch_first
            cells[ time_step, :, : ] = states[ 1 ]
        
        # cell: Batch x seq_len x hidden_size
        cells = cells.transpose( 0, 1 )

        # Data parallelism for adaptive attention block
        if torch.cuda.device_count() > 1:
            ids = range( torch.cuda.device_count() )
            adaptive_block_parallel = nn.DataParallel( self.adaptive, device_ids=ids )
            
            scores, atten_weights, beta = adaptive_block_parallel( x, hiddens, cells, V )
        else:
            scores, atten_weights, beta = self.adaptive( x, hiddens, cells, V )
        
        # print("!!!!!STATE SIZE:", np.shape(states))
        
        # Return states for Caption Sampling purpose
        return scores, states, atten_weights, beta
    
        

# Whole Architecture with Image Encoder and Caption decoder        
class Encoder2Decoder( nn.Module ):
    def __init__( self, embed_size, vocab_size, hidden_size ):
        super( Encoder2Decoder, self ).__init__()
        
        # Image CNN encoder and Adaptive Attention Decoder
        self.encoder = AttentiveCNN( embed_size, hidden_size )
        self.decoder = Decoder( embed_size, vocab_size, hidden_size )
        
        
    def forward( self, images, captions, lengths ):
        
        # Data parallelism for V v_g encoder if multiple GPUs are available
        # V=[ v_1, ..., v_k ], v_g in the original paper
        if torch.cuda.device_count() > 1:
            device_ids = range( torch.cuda.device_count() )
            encoder_parallel = torch.nn.DataParallel( self.encoder, device_ids=device_ids )
            V, v_g = encoder_parallel( images ) 
        else:
            V, v_g = self.encoder( images )
        
        # Language Modeling on word prediction
        scores, _, _,_ = self.decoder( V, v_g, captions )
        
        # Pack it to make criterion calculation more efficient
        packed_scores = pack_padded_sequence( scores, lengths, batch_first=True )
        
        return packed_scores
    
    # Caption generator
    def sampler( self, images, max_len=20 ):
        """
        Samples captions for given image features (Greedy search).
        """
        
        # Data parallelism if multiple GPUs
        if torch.cuda.device_count() > 1:
            device_ids = range( torch.cuda.device_count() )
            encoder_parallel = torch.nn.DataParallel( self.encoder, device_ids=device_ids )
            V, v_g = encoder_parallel( images ) 
        else:    
            V, v_g = self.encoder( images )
            
        # Build the starting token Variable <start> (index 1): B x 1
        if torch.cuda.is_available():
            captions = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ).cuda() )
        else:
            captions = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ) )
        
        # Get generated caption idx list, attention weights and sentinel score
        sampled_ids = []
        attention = []
        Beta = []
        
        # Initial hidden states
        states = None

        for i in range( max_len ):

            scores, states, atten_weights, beta = self.decoder( V, v_g, captions, states ) 
            predicted = scores.max( 2 )[ 1 ] # argmax
            captions = predicted
            
            # Save sampled word, attention map and sentinel at each timestep
            sampled_ids.append( captions )
            attention.append( atten_weights )
            Beta.append( beta )
        
        # caption: B x max_len
        # attention: B x max_len x 49
        # sentinel: B x max_len
        sampled_ids = torch.cat( sampled_ids, dim=1 )
        attention = torch.cat( attention, dim=1 )
        Beta = torch.cat( Beta, dim=1 )
        
        return sampled_ids, attention, Beta

    def mysampler( self, images, max_len=20, beam_size=3, decay=1 ):
        """
        Samples captions for given image features (Greedy search).
        """
        
        # Data parallelism if multiple GPUs
        if torch.cuda.device_count() > 1:
            device_ids = range( torch.cuda.device_count() )
            encoder_parallel = torch.nn.DataParallel( self.encoder, device_ids=device_ids )
            V, v_g = encoder_parallel( images ) 
        else:    
            V, v_g = self.encoder( images )
            
        # Build the starting token Variable <start> (index 1): B x 1
        if torch.cuda.is_available():
            captions = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ).cuda())
        else:
            captions = Variable( torch.LongTensor( images.size( 0 ), 1 ).fill_( 1 ) )
        

        # Get generated caption idx list, attention weights and sentinel score
        sampled_ids = []
        attention = []
        Beta = []
        # Initial hidden states
        states = None

#         beam_size = 7
        word_pred_list = []
        state_list = []
        batch_size = images.size( 0 )


        for i in range(beam_size):
            word_pred_list.append([[[1],1]]*batch_size)
            state_list.append(states)




        for i in range( max_len ):
            newstate_list = []
            logits_list = []
            tmpatten = None
            tmpbeta = None
            
            # lengths of sequence(non-padding) for each beam
            length_list = np.zeros(beam_size)
            
            for j in range(beam_size):
                captions = [each[0][-1] for each in word_pred_list[j]]
                # print "enter 0"
                if torch.cuda.is_available():
                    # print "enter 1"
                    # print len(captions)
                    # print len(captions[0])
                    captions =  Variable(torch.LongTensor( np.reshape(captions,(-1,1)) )).cuda()
                else:
                    captions =  Variable(torch.LongTensor( np.reshape(captions,(-1,1)) ))
                # print captions.size()
                # print "entered"
                scores, states, atten_weights, beta = self.decoder( V, v_g, captions, state_list[j] )
                # print "passed"
                newstate_list.append(states) 
                tmpatten =atten_weights
                tmpbeta = beta
                
                scores = scores.view( (batch_size, -1) )
                #scores = Variable( scores )
                scores = torch.nn.functional.softmax(scores, dim=1)
                
                # sum of log(prob)
#                 scores = (scores.data.cpu().numpy())
                scores = np.log(scores.data.cpu().numpy())

                # print("PREVIOUS Score", scores[0, 0])

                prescore = []
                for i in range(batch_size):
                    prescore.append(word_pred_list[j][i][1])
                    
                scores += decay * (np.reshape(prescore, [batch_size,1]))
                # scores = scores * (np.reshape(prescore, [batch_size,1]) + 1)
                logits_list.append(scores)
                
                # print("AFTER Score", scores[0, 0])

            Vs = np.shape(logits_list[0])[1]
            if i==0:
                logits_list = np.array(logits_list[0])
            else:
                logits_list = np.array(logits_list).transpose([1,0,2]).reshape([batch_size,-1])
            logits_pick = np.argsort(logits_list)[:,-beam_size:]
            new_word_pred_list = [[] for _  in range(beam_size)]

            
            for k in range(beam_size):
                
                for j in range(batch_size):
                    idx = logits_pick[j][k]
                    #print "idx",idx
                    whichbeam = int(idx/Vs)
                    #print "whichbeam",whichbeam
                    # print("STATE LIST SHAPE", np.shape(state_list))
#                     print("STATE LIST SHAPE", state_list[0])
#                     print("NEWSTATE LIST SHAPE", np.shape(newstate_list))
                    # print("NEWSTATE LIST SHAPE", (newstate_list[0]))
                    
                    state_list[k] = newstate_list[whichbeam]
#                     state_list[k][:][0][j] = newstate_list[whichbeam][:][0][j]
#                     state_list[i][0][j]=newstate_list[whichbeam][0][j]
#                     state_list[i][1][j]=newstate_list[whichbeam][1][j]
                    prev = word_pred_list[whichbeam][j]
                    #print "prev",prev
                    cur_word = idx%Vs
                    # print "cur_word", cur_word
                    cur_prob = logits_list[j][idx]
                    tmp = copy.deepcopy(prev[0])
                    tmp.append(cur_word)
                    cur = [tmp, cur_prob]
                    #print "cur",cur
                    new_word_pred_list[k].append(cur)
            word_pred_list = new_word_pred_list





            attention.append( tmpatten )
            Beta.append( tmpbeta )
        
        # caption: B x max_len
        # attention: B x max_len x 49
        # sentinel: B x max_len
        sampled_captions = []
        for i in range(batch_size):
            tmplist = []
            for j in range(beam_size):
                tmplist.append(word_pred_list[j][i])
            tmplist = sorted(tmplist, reverse=False, key=lambda l: l[1])
            #print "put in the cap", tmplist[-1][0]
            sampled_captions.append(tmplist[-1][0][1:])
            
        sampled_ids = Variable( torch.LongTensor(sampled_captions).view((batch_size, -1)) )
        attention = torch.cat( attention, dim=1 )
        Beta = torch.cat( Beta, dim=1 )
        
        return sampled_ids, attention, Beta