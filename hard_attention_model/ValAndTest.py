import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import pickle
from utils import CocoImageFolder, to_var, show_images
from adaptive import Encoder2Decoder
from build_vocab import Vocabulary
from torch.autograd import Variable
from torchvision import transforms

class EvalValTest:

    def __init__(self, args):
        self.args = args
        self.pretrained = "./models/" + args.pretrained
        self.num_workers = 4
        self.eval_size = args.eval_size
        self.beam_size = args.beam_size
        self.vocab_path = './data/vocab.pkl'
        self.vocab = None
        self.transform = None
        self.model = None




    def main(self):
        print "********************Overhead Operations***************************"

        with open(self.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        # Image transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        # Load model
        print "Loading model {}......".format(self.pretrained)
        self.model = Encoder2Decoder(256, len(self.vocab), 512)
        self.model.load_state_dict(torch.load(self.pretrained))
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        print "Model loaded!"

        print "********************Validation Phase***************************"

        images_path = './data/resized/val2014/'
        caption_path = './data/annotations/captions_val2014.json'

        self.eval(images_path, caption_path, self.args.val_saved_name)

        print "********************Test Phase***************************"

        images_path = './data/resized/test2014/'
        caption_path = './data/annotations/image_info_test2014.json'

        self.eval(images_path, caption_path, self.args.test_saved_name)


    def eval(self, images_path, caption_path, name):
        eval_size = self.args.eval_size
        beam_size = self.args.beam_size

        # Get image data loader
        cocoFolder = CocoImageFolder(images_path, caption_path, self.transform)
        data_loader = torch.utils.data.DataLoader(
            cocoFolder,
            batch_size=self.eval_size,
            shuffle=False, num_workers=self.num_workers,
            drop_last=False)

        num_batches = len(data_loader)
        res = []
        # every item in list is a batch of imgs, imgids, filenames
        for i, (images, image_ids, filenames) in enumerate(data_loader):
            if i % 100 == 0:
                print "Processed {}/{}".format(i, num_batches)

            images = to_var(images)
            # generated_captions, attention, beta = self.model.sampler( images )
            # with beam search
            generated_captions, attention, beta = self.model.mysampler(images, beam_size=self.beam_size)

            captions = generated_captions.cpu().data.numpy()

            for image_idx in range(captions.shape[0]):

                sampled_ids = captions[image_idx]
                sampled_caption = []

                for word_id in sampled_ids:

                    word = self.vocab.idx2word[word_id]
                    if word == '<end>':
                        break
                    else:
                        sampled_caption.append(word)

                sentence = ' '.join(sampled_caption)

                res.append({"image_id": image_ids[image_idx], "caption": sentence})

        # save results to file
        resName = "./results/" + name + ".json"
        with open(resName, 'w') as file:
            json.dump(res, file)
        print "{} is saved!".format(resName)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="adaptive-17.pkl",
                        help="Pretrained model in ./models/")
    parser.add_argument("--eval_size", type=int, default=20, help="Evalution batch size")
    parser.add_argument("--val_saved_name", type=str, default="captions_val2014__results",
                        help="Saved val json file name")
    parser.add_argument("--test_saved_name", type=str, default="captions_test2014__results",
                        help="Saved test json file name")
    parser.add_argument("--beam_size", type=int, default=3, help="beam search size")
    args = parser.parse_args()

    print "******************Evaluation Settings********************"
    print(args)

    EvalValTest(args).main()