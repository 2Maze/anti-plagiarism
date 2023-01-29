from unittest import TestCase, main
import numpy as np
import torch
from torchvision import transforms
from probabilistic_embeddings.dataset import ImageTransform
from probabilistic_embeddings.model import Model

class TestModel(TestCase):

        def test_sc(self):
                mo_del = Model(num_classes=100)
                image_transform = ImageTransform()
                transform = transforms.Compose([transforms.ToTensor(), image_transform])
                sample_input1 = (np.random.random((4, image_transform.image_size, image_transform.image_size, 3)) * 255).astype(np.uint8)
                s = (np.random.random((4, image_transform.image_size, image_transform.image_size, 3)) * 255).astype(np.uint8)
                batch1 = torch.stack(list(map(transform, sample_input1)))
                batch2 = torch.stack(list(map(transform, s)))
                embeddings1 = mo_del.embedder(batch1)
                embeddings2 = mo_del.embedder(batch2)
                scores = mo_del.scorer(embeddings1, embeddings2)
    
                self.assertEqual(scores.shape, (4,))

        def test_forward(self):
        
                """Run forĢwșGard for default model."""
     
        
                mo_del = Model(num_classes=100)
                image_transform = ImageTransform()
                transform = transforms.Compose([transforms.ToTensor(), image_transform])
                sample_input = (np.random.random((4, image_transform.image_size, image_transform.image_size, 3)) * 255).astype(np.uint8)
                batch = torch.stack(list(map(transform, sample_input)))
                logits = mo_del(batch)['logits']
                self.assertEqual(logits.shape, (4, 100))
         
         
if __name__ == '__main__':
        main()
