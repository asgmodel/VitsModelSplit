
import numpy as np
import os
from datasets import Dataset,DatasetDict
from typing import Union,List,Dict
import torch
from dataclasses import dataclass
from transformers.feature_extraction_utils import BatchFeature
from VitsModelSplit.feature_extraction import VitsFeatureExtractor
from VitsModelSplit.vits_model import VitsModel
from transformers import AutoTokenizer

#.............................................


@dataclass
class DataSetFeaturesCollector:

    def __init__(self,tokenizer,model,feature_extractor,forward_attention_mask=True) -> None:
        self.tokenizer=tokenizer
        self.feature_extractor = feature_extractor
        self.model=model
        self.forward_attention_mask = forward_attention_mask

    #.............................................

    def pad_waveform(self, raw_speech):
        
        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        batched_speech = BatchFeature({"input_features": raw_speech})

        # convert into correct format for padding

        padded_inputs = self.feature_extractor.pad(
            batched_speech,
            padding=True,
            return_attention_mask=False,
            return_tensors="pt",
        )["input_features"]

        return padded_inputs
    
    #.............................................
    
    def prepare_dataset(self,batch):
        
        sample = batch['audio']
        audio_inputs = self.feature_extractor(
                sample,
                sampling_rate=16000,
                return_attention_mask=False,
                do_normalize=False,
            )

        batch["labels"] = audio_inputs.get("input_features")[0]
        batch["waveform_input_length"] = len(sample)
        batch["waveform"] = batch['audio']
        batch["mel_scaled_input_features"] = audio_inputs.get("mel_scaled_input_features")[0]
        textsample = batch['text']
        inputs = self.tokenizer(textsample, return_tensors="pt")
        inputs = self.tokenizer.pad({'input_ids':inputs.input_ids})
        batch['input_ids'] = inputs.input_ids
        batch['attention_mask'] = inputs.attention_mask
       # batch['speaker_id']=batch['speaker_id']


        return batch
    
    
    #.............................................


    def __call__(self, dataset: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods

        dataset = Dataset.from_list(dataset)
        features = dataset.map(
                self.prepare_dataset,
                remove_columns=dataset.column_names,
                desc="preprocess",
                )

        features = list(features)
     
        model_input_name = "input_ids"
        
        input_ids = [{model_input_name: feature[model_input_name][0]} for feature in features]
        
        # pad input tokens
        batch = self.tokenizer.pad(input_ids, return_tensors="pt", return_attention_mask=self.forward_attention_mask)
   
        # pad waveform
        waveforms = [np.array(feature["waveform"]) for feature in features]
        batch["waveform"] = self.pad_waveform(waveforms)

        # pad spectrogram
        label_features = [np.array(feature["labels"]) for feature in features]
        labels_batch = self.feature_extractor.pad(
            {"input_features": [i.T for i in label_features]}, return_tensors="pt", return_attention_mask=True
        )

        labels = labels_batch["input_features"].transpose(1, 2)
        batch["labels"] = labels
        batch["labels_attention_mask"] = labels_batch["attention_mask"]

        # pad mel spectrogram
        mel_scaled_input_features = {
            "input_features": [np.array(feature["mel_scaled_input_features"]).squeeze().T for feature in features]
        }
        mel_scaled_input_features = self.feature_extractor.pad(
            mel_scaled_input_features, return_tensors="pt", return_attention_mask=True
        )["input_features"].transpose(1, 2)

        batch["mel_scaled_input_features"] = mel_scaled_input_features
        batch["speaker_id"] = (
            torch.tensor([feature["speaker_id"] for feature in dataset]) if "speaker_id" in dataset[0] else None
        )
        
        # with torch.no_grad():
        #     padding_mask  =torch.ones_like(batch['input_ids']).unsqueeze(-1).float()
        #     text_encoder_output = self.model.text_encoder(batch['input_ids'],
        #                                                 padding_mask=padding_mask,
        #                                                 attention_mask = batch['attention_mask']
        #                                                 )
        #     batch['text_encoder_output'] = text_encoder_output 
        #     posterior_latents, posterior_means, posterior_log_variances = self.model.posterior_encoder(
        #             batch['labels'], batch['labels_attention_mask'].unsqueeze(1).float()
        #            )
        #     posterior_encode_output={
        #       'posterior_latents':posterior_latents,
        #       'posterior_means':posterior_means,
        #       'posterior_log_variances':posterior_log_variances
        #     }
        #     batch['posterior_encode_output']=posterior_encode_output


        
        return batch


#..............................................................



#.............................................

def run_dataset_features_collection(
                         dataset_dir,
                         train_split_name ="train",
                         eval_split_name="eval",
                         full_generation_name = 'full_generation',
                         tokenizer = None,
                         model = None,
                         feature_extractor = None,
                         train_batch_size = 1,
                         eval_batch_size = 1,
                         output_dir = "dataset_features"
                         
                         ):
    
    dataset = DatasetDict.load_from_disk(dataset_dir)
    
    data_collator = DataSetFeaturesCollector(
        tokenizer = tokenizer,
        model = model,
        feature_extractor = feature_extractor,
        forward_attention_mask = True
        )
    
    if train_split_name:
        train_dataloader = torch.utils.data.DataLoader(
            dataset[train_split_name],
            shuffle=False,
            collate_fn=data_collator,
            batch_size=train_batch_size,
            sampler=None,
        )
        
        train_dir = os.path.join(output_dir,"train")
        os.makedirs(train_dir,exist_ok=True)
        
        for step, batch in enumerate(train_dataloader):
            print(f"Train Dataset - batch {step}, waveform {(batch['waveform'].shape)},tokens {(batch['input_ids'].shape)}... ")
            fname = os.path.join(train_dir,f"train-batch-{step}.bin")
            with open(fname, "wb") as f:
                torch.save(batch, f)
            
    if eval_split_name:
        
        eval_dataloader = torch.utils.data.DataLoader(
            dataset[eval_split_name],
            shuffle=False,
            collate_fn=data_collator,
            batch_size=eval_batch_size,
            sampler=None,
        )
        
        eval_dir = os.path.join(output_dir,"eval")
        os.makedirs(eval_dir,exist_ok=True)
        
        for step, batch in enumerate(eval_dataloader):
            print(f"Eval Dataset - batch {step}, waveform {(batch['waveform'].shape)},tokens {(batch['input_ids'].shape)}... ")
            fname = os.path.join(eval_dir,f"eval-batch-{step}.bin")
            with open(fname, "wb") as f:
                torch.save(batch, f)  
    
    if full_generation_name:
        
        full_generation_dataloader = torch.utils.data.DataLoader(
            dataset[full_generation_name],
            shuffle=False,
            collate_fn=data_collator,
            batch_size=1,
            sampler=None,
        )
        
        full_generation_dir = os.path.join(output_dir,"full_generation")
        os.makedirs(full_generation_dir,exist_ok=True)
        
        for step, batch in enumerate(full_generation_dataloader):
            print(f"Full Generation Dataset - batch {step}, waveform {(batch['waveform'].shape)},tokens {(batch['input_ids'].shape)}... ")
            fname = os.path.join(full_generation_dir,f"full-generation-batch-{step}.bin")
            with open(fname, "wb") as f:
                torch.save(batch, f)  

#...........................................................................

import torch.utils.data 

class FeaturesCollectionDataset(torch.utils.data.Dataset):
    
    def __init__(self,dataset_dir,device='cpu') -> None:
        self.dataset_dir = dataset_dir
        self.batchs_path = sorted([os.path.join(self.dataset_dir,file) for file in os.listdir(dataset_dir) if file.endswith('.bin')])
        self.device = device
        
    def __len__(self):
        return len(self.batchs_path)
    
    def __getitem__(self, idx):
        batch_name = self.batchs_path[idx]
        with open(batch_name, "rb") as f:
            batch = torch.load(f,map_location=torch.device(self.device))
        return batch
        
        
class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths =dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
class VitsCollectionDataset(torch.utils.data.Dataset):
    
    def __init__(self,dataset,hop_length=256,rate=16_000,device='cpu') -> None:
        self.dataset = dataset
        self.lengths =(torch.tensor(dataset['secs'])*rate//(2*hop_length)).tolist()
        self.device = device


        
    def __len__(self):
        return self.dataset.num_rows
     
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def  get_dataloader(dir_db_train,feature_extractor,name_db='train',batch_size=8,num_workers=0):
    dataset = DatasetDict.load_from_disk(dir_db_train)
    db_train=VitsCollectionDataset(dataset[name_db])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=VitsModel.from_pretrained("facebook/mms-tts-ara").to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ara",cache_dir="./")#.to("cuda")
    train_sampler = DistributedBucketSampler(
          db_train,
          batch_size,
          [32,300,400,500,600,700,800,900,1000],
          num_replicas=1,
          rank=0,
          shuffle=True)
    data_collator = DataSetFeaturesCollector(
        tokenizer = tokenizer,
        model = model,
        feature_extractor = feature_extractor,
        forward_attention_mask = True
        )
    train_dataloader = torch.utils.data.DataLoader(
              db_train,
              num_workers=num_workers, shuffle=False, pin_memory=True,
          collate_fn=data_collator, batch_sampler=train_sampler
            )
    return train_dataloader   
