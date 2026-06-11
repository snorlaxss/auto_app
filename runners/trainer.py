import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tqdm import tqdm


# from datasets.datasets_nocs import get_data_loaders_from_cfg, process_batch
from datasets.datasets_omni6dpose import get_data_loaders_from_cfg, process_batch
from networks.posenet_agent import PoseNet 
from configs.config import get_config
from utils.transforms import *
from cutoop.transform import *
from cutoop.data_types import *
from cutoop.eval_utils import *

def train_score(cfg, train_loader, val_loader, test_loader, score_agent, teacher_model=None):
    """ Train score network or energe network without ranking
    Args:
        cfg (dict): config file
        train_loader (torch.utils.data.DataLoader): train dataloader
        val_loader (torch.utils.data.DataLoader): validation dataloader
        score_agent (torch.nn.Module): score network or energy network without ranking
    Returns:
    """
    
    for epoch in range(score_agent.clock.epoch, cfg.n_epochs):
        ''' train '''
        torch.cuda.empty_cache()
        print(f"[train_score] Starting epoch {epoch} - preparing data iterator")
        pbar = tqdm(train_loader)
        successful_batches = 0
        print("[train_score] Entered training loop")
        for i, batch_sample in enumerate(pbar):
            try:
                ''' warm up'''
                if score_agent.clock.step < cfg.warmup:
                    score_agent.update_learning_rate()
                    
                ''' load data '''
                batch_sample = process_batch(
                    batch_sample = batch_sample, 
                    device=cfg.device, 
                    pose_mode=cfg.pose_mode, 
                    PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS, 
                )
                
                ''' train score or energe without feedback'''
                losses = score_agent.train_func(data=batch_sample, gf_mode='score', teacher_model=teacher_model)
                
                pbar.set_description(f"EPOCH_{epoch}[{i}/{len(pbar)}][loss: {[value.item() for key, value in losses.items()]}][successful: {successful_batches}]")
                score_agent.clock.tick()
                successful_batches += 1
                
            except (RecursionError, RuntimeError, Exception) as e:
                print(f"\nError processing batch {i} in epoch {epoch}: {str(e)[:100]}...")
                print("Skipping this batch and continuing...")
                continue
        
        print(f"Epoch {epoch}: Successfully processed {successful_batches}/{len(pbar)} batches")
        
        ''' updata learning rate and clock '''
        # if epoch >= 50 and epoch % 50 == 0:
        score_agent.update_learning_rate()
        score_agent.clock.tock()

        ''' start eval '''
        if score_agent.clock.epoch % cfg.eval_freq == 0:   
            data_loaders = [train_loader, val_loader, test_loader]    
            data_modes = ['train', 'val', 'test']   
            for i in range(len(data_modes)):
                try:
                    test_batch = next(iter(data_loaders[i]))
                    data_mode = data_modes[i]
                    test_batch = process_batch(
                        batch_sample=test_batch,
                        device=cfg.device,
                        pose_mode=cfg.pose_mode,
                    )
                    score_agent.eval_func(test_batch, data_mode)
                except Exception as e:
                    print(f"Error in evaluation for {data_modes[i]}: {e}")
                    continue
                
            ''' save (ema) model '''
            score_agent.save_ckpt()



def train_scale(cfg, train_loader, val_loader, test_loader, scale_agent, score_agent):
    """ Train scale network
    Args:
        cfg (dict): config file
        train_loader (torch.utils.data.DataLoader): train dataloader
        val_loader (torch.utils.data.DataLoader): validation dataloader
        scale_agent (torch.nn.Module): scale network
        score_agent (torch.nn.Module): score network
    Returns:
    """

    score_agent.eval()
    
    for epoch in range(score_agent.clock.epoch, cfg.n_epochs):
        ''' train '''
        torch.cuda.empty_cache()
        # For each batch in the dataloader
        pbar = tqdm(train_loader)
        for i, batch_sample in enumerate(pbar):
            
            ''' warm up'''
            if score_agent.clock.step < cfg.warmup:
                score_agent.update_learning_rate()
                
            ''' load data '''
            batch_sample = process_batch(
                batch_sample = batch_sample, 
                device=cfg.device, 
                pose_mode=cfg.pose_mode, 
                PTS_AUG_PARAMS=cfg.PTS_AUG_PARAMS, 
            )
            
            ''' train scale'''
            with torch.no_grad():
                score_agent.encode_func(data=batch_sample)
            losses = scale_agent.train_func(data=batch_sample, gf_mode='scale')
            
            pbar.set_description(f"EPOCH_{epoch}[{i}/{len(pbar)}][loss: {[value.item() for key, value in losses.items()]}]")
            scale_agent.clock.tick()
        
        ''' updata learning rate and clock '''
        # if epoch >= 50 and epoch % 50 == 0:
        scale_agent.update_learning_rate()
        scale_agent.clock.tock()

        ''' start eval '''
        if scale_agent.clock.epoch % cfg.eval_freq == 0:   
            data_loaders = [train_loader, val_loader, test_loader]    
            data_modes = ['train', 'val', 'test']   
            for i in range(len(data_modes)):
                test_batch = next(iter(data_loaders[i]))
                data_mode = data_modes[i]
                test_batch = process_batch(
                    batch_sample=test_batch,
                    device=cfg.device,
                    pose_mode=cfg.pose_mode,
                )
                with torch.no_grad():
                    score_agent.encode_func(data=test_batch)
                scale_agent.eval_func(test_batch, data_mode, gf_mode='scale')
                
            ''' save (ema) model '''
            scale_agent.save_ckpt()

def main():
    # load config
    cfg = get_config()
    print("[main] Loaded config. agent_type:", cfg.agent_type); sys.stdout.flush()
    
    ''' Init data loader '''
    if not (cfg.eval or cfg.pred):
        data_loaders = get_data_loaders_from_cfg(cfg=cfg, data_type=['train', 'val', 'test'])
        train_loader = data_loaders['train_loader']
        val_loader = data_loaders['val_loader']
        test_loader = data_loaders['test_loader']
        print('train_set: ', len(train_loader))
        print('val_set: ', len(val_loader))
        print('test_set: ', len(test_loader))
    else:
        data_loaders = get_data_loaders_from_cfg(cfg=cfg, data_type=['test'])
        test_loader = data_loaders['test_loader']   
        print('test_set: ', len(test_loader))
    sys.stdout.flush()
  
    
    ''' Init trianing agent and load checkpoints'''
    if cfg.agent_type == 'score':
        print("[main] Creating score_agent..."); sys.stdout.flush()
        cfg.agent_type = 'score'
        score_agent = PoseNet(cfg)
        ##
        if cfg.pretrained_score_model_path is not None:
            score_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
        ##
        print("[main] score_agent created"); sys.stdout.flush()
        tr_agent = score_agent

    elif cfg.agent_type == 'scale':
        cfg.agent_type = 'score'
        score_agent = PoseNet(cfg)
        score_agent.load_ckpt(model_dir=cfg.pretrained_score_model_path, model_path=True, load_model_only=True)
        cfg.agent_type = 'scale'
        scale_agent = PoseNet(cfg)
        tr_agent = scale_agent
    else:
        raise NotImplementedError
    
    ''' Load checkpoints '''
    if cfg.use_pretrain or cfg.eval or cfg.pred:
        print("[main] Loading checkpoint for tr_agent...", flush=True)
        sys.stdout.flush()
        tr_agent.load_ckpt(
            model_dir=(
                cfg.pretrained_score_model_path if cfg.agent_type == 'score' else (
                    cfg.pretrained_scale_model_path
                )
            ), 
            model_path=True, 
            load_model_only=False
        )
        print("[main] Finished checkpoint load (if any)"); sys.stdout.flush()
                
        
    ''' Start training loop '''
    print("[main] About to start training loop with agent_type:", cfg.agent_type); sys.stdout.flush()
    if cfg.agent_type == 'score':
        train_score(cfg, train_loader, val_loader, test_loader, tr_agent)
    elif cfg.agent_type == 'scale':
        train_scale(cfg, train_loader, val_loader, test_loader, tr_agent, score_agent)
    else:
        raise NotImplementedError(f"Unsupported agent_type: {cfg.agent_type}")
if __name__ == '__main__':
    main()


