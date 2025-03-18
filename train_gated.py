import torch
import numpy as np
import os
import sys
import math
import logging
import json
from tqdm import tqdm, trange
from process_eicu_k_fold import get_datasets
from utils import *
from graph_convolutional_transformer import GraphConvolutionalTransformer
from datetime import datetime
import pandas as pd

from tensorboardX import SummaryWriter
import torchsummary as summary

    
def prediction_loop(args, model, dataloader, priors_datalaoders, description='Evaluating'):

    batch_size = dataloader.batch_size
    eval_losses = []
    preds = None
    label_ids = None
    model.eval()
    priors_datalaoder, bert_priors_datalaoder, kg_priors_datalaoder  = priors_datalaoders
    for data, priors_data, bert_priors_data, kg_priors_data in tqdm(zip(dataloader, priors_datalaoder, bert_priors_datalaoder, kg_priors_datalaoder), desc=description, ncols=10, mininterval=10):

        data, priors_data = prepare_data(data, priors_data, args.device)
        # 
        bert_priors = {}
        bert_priors['indices'] = bert_priors_data[0].to(args.device)
        bert_priors['values'] = bert_priors_data[1].to(args.device)
        kg_priors = {}
        kg_priors['indices'] = kg_priors_data[0].to(args.device)
        kg_priors['values'] = kg_priors_data[1].to(args.device)
        all_priors = (priors_data, bert_priors, kg_priors)    
        with torch.no_grad():
            outputs = model(data, all_priors)
            loss = outputs[0].mean().item()
            logits = outputs[1]
        
        labels = data[args.label_key]
        
        batch_size = data[list(data.keys())[0]].shape[0]
        eval_losses.extend([loss]*batch_size)
        preds = logits if preds is None else nested_concat(preds, logits, dim=0)
        label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)
    
    if preds is not None:
        preds = nested_numpify(preds)
    if label_ids is not None:
        label_ids = nested_numpify(label_ids)
    metrics = compute_metrics(preds, label_ids)
    
    metrics['eval_loss'] = np.mean(eval_losses)
    
    for key in list(metrics.keys()):
        if not key.startswith('eval_'):
            metrics['eval_{}'.format(key)] = metrics.pop(key)
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return list(np.exp(x) / np.sum(np.exp(x), axis=0))
    probs_preds = np.hstack((np.array(list(map(softmax, list(preds)))), np.argmax(preds, axis=1).reshape(-1, 1)))
    preds_and_label_ids = np.hstack((probs_preds, label_ids.reshape(-1, 1)))
    return metrics, preds_and_label_ids

def is_save_model(best_results, metrics):
    suffixes = []
    is_save = False
    if 'eval_AUROC' in metrics:
        if (best_results is None):
            best_results = metrics
            return True, ['overall_best'], best_results
        else:
            if (best_results['eval_AUROC'] < metrics['eval_AUROC']) and (best_results['eval_AUCPR'] < metrics['eval_AUCPR']):
                best_results['eval_AUROC'] = metrics['eval_AUROC']
                best_results['eval_AUCPR'] = metrics['eval_AUCPR']
                is_save = True
                suffixes.append('overall_best')
            if (best_results['eval_AUROC'] <= metrics['eval_AUROC']):
                best_results['eval_AUROC'] = metrics['eval_AUROC']
                is_save = True
                suffixes.append('roc_best')
            if (best_results['eval_AUCPR'] <= metrics['eval_AUCPR']):
                best_results['eval_AUCPR'] = metrics['eval_AUCPR']
                is_save = True
                suffixes.append('pr_best')
            if (best_results['eval_loss'] > metrics['eval_loss']):
                best_results['eval_loss'] = metrics['eval_loss']
                is_save = True
                suffixes.append('loss_best')
            return is_save, suffixes, best_results
    else:
        if (best_results is None):
            best_results = metrics
            return True, ['acc_best'], best_results
        elif (best_results['eval_ACC'] <= metrics['eval_ACC']):
            best_results['eval_ACC'] = metrics['eval_ACC']
            is_save = True
            suffixes.append('acc_best')
        return is_save, suffixes, best_results


    


def main():
    args = ArgParser().parse_args()
    sub_dir = '_0912_gated'
    set_seed(args.seed) 
    if not os.path.exists(os.path.dirname(args.log_dir)):
        os.mkdir(os.path.dirname(args.log_dir))

    logging.basicConfig(
        filename=args.log_dir,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    logger = logging.getLogger(__name__)
    
    logging.info('*********************task description************************')
    logging.info(args.task_desc)
    logging.info("Arguments %s", args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging_dir = os.path.join(args.output_dir, 'logging')
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    log_sub_dir = f"{sub_dir}" # datetime.now().strftime("_%m%d")
    log_sub_dir = os.path.join(log_sub_dir, f"{args.task_desc}_{args.fold}")
    log_dir = os.path.join(logging_dir, log_sub_dir)
    tb_writer = SummaryWriter(log_dir=log_dir, filename_suffix=f"{args.task_desc}_{args.fold}")

    # Dataset handling
    datasets, prior_guides, bert_prior_guides, kg_prior_guides = get_datasets(args.data_dir, args.prior_type, fold=args.fold)
    train_dataset, eval_dataset, test_dataset = datasets
    train_priors, eval_priors, test_priors = prior_guides
    # print(train_dataset.__len__())
    # 
    bert_train_priors, bert_eval_priors, bert_test_priors = bert_prior_guides
    kg_train_priors, kg_eval_priors, kg_test_priors = kg_prior_guides

    train_priors_dataset = eICUDataset(train_priors)
    eval_priors_dataset = eICUDataset(eval_priors)
    test_priors_dataset = eICUDataset(test_priors)

    # 
    bert_train_priors_dataset = eICUDataset(bert_train_priors)
    bert_eval_priors_dataset = eICUDataset(bert_eval_priors)
    bert_test_priors_dataset = eICUDataset(bert_test_priors)
    kg_train_priors_dataset = eICUDataset(kg_train_priors)
    kg_eval_priors_dataset = eICUDataset(kg_eval_priors)
    kg_test_priors_dataset = eICUDataset(kg_test_priors)

    if args.mdp:
        data_collate_fn = data_collate_fn_for_mdp
    else:
        # data_collate_fn = None
        # if args.label_key == 'los':
        #     data_collate_fn = data_collate_fn_for_los
        data_collate_fn = data_collate_fn_for_los
        
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn)
    
    print("data_loader over!")


    train_priors_dataloader = DataLoader(train_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    eval_priors_dataloader = DataLoader(eval_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    test_priors_dataloader = DataLoader(test_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    
    # 
    bert_train_priors_dataloader = DataLoader(bert_train_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    bert_eval_priors_dataloader = DataLoader(bert_eval_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    bert_test_priors_dataloader = DataLoader(bert_test_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    kg_train_priors_dataloader = DataLoader(kg_train_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    kg_eval_priors_dataloader = DataLoader(kg_eval_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    kg_test_priors_dataloader = DataLoader(kg_test_priors_dataset, batch_size=args.batch_size, collate_fn=priors_collate_fn)
    
    

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.device.type == 'cuda':
        torch.cuda.set_device(args.device)
    print(args.device)
    args.device = torch.device('cuda:0')
    saved_model_dir = os.path.join(os.getcwd(), f'saved_models/{sub_dir}/')
    if args.do_train:
        model = GraphConvolutionalTransformer(args)

        if args.do_prompt:
            assert args.label_key !="mdp_base" , f'prompt is used for expired/readmission predition task, but now doing {args.label_key} prediction task' 
            # model_type = "mdp_base"
            model_type = "los_base"
            # if args.use_adr_pooler or args.do_gate_mechanism:
            #     model_type = f"mdp_{args.task_desc.split('_')[-1]}"
            if args.num_stacks == 2:
                temp_model = torch.load(os.path.join(os.getcwd(), f'saved_models/gated/{model_type}_{args.fold}_final.pth'))
            elif args.num_stacks == 3:
                temp_model = torch.load(os.path.join(os.getcwd(), f'saved_models/gated/{model_type}_{args.fold}_final.pth'))            
            loaded_dict = {}
            for k, v in temp_model.state_dict().items():
                if (not 'classifier' in k) and (not 'embeddings' in k):
                    loaded_dict[k] = v
            model.load_state_dict(loaded_dict, strict=False)

        model = model.to(args.device)
        
        num_update_steps_per_epoch = len(train_dataloader)
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(args.max_steps % num_update_steps_per_epoch > 0)
        else:
            max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
            num_train_epochs = args.num_train_epochs
        num_train_epochs = int(np.ceil(num_train_epochs))
        
        args.eval_steps = num_update_steps_per_epoch // 2

        parameters = model.parameters()

        optimizer = torch.optim.Adamax(parameters, lr=args.learning_rate)
        
        warmup_steps = max_steps // (1 / args.warmup)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps=max_steps)
        
        logger.info('***** Running Training *****')
        logger.info(' Num examples = {}'.format(len(train_dataloader.dataset)))
        logger.info(' Num epochs = {}'.format(num_train_epochs))
        logger.info(' Train batch size = {}'.format(args.batch_size))
        logger.info(' Total optimization steps = {}'.format(max_steps))

        epochs_trained = 0
        global_step = 0
        tr_loss = torch.tensor(0.0).to(args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        best_results = None
        selected_key = 'eval_ACC' if args.mdp else 'eval_loss'
        train_pbar = trange(epochs_trained, num_train_epochs, desc='Epoch')

        print(args.logging_steps, args.eval_steps, num_update_steps_per_epoch)

        early_stop_steps = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_pbar = train_dataloader
            for data, priors_data, bert_priors_data, kg_priors_data in zip(train_dataloader, train_priors_dataloader, bert_train_priors_dataloader, kg_train_priors_dataloader):
                model.train()
                data, priors_data = prepare_data(data, priors_data, args.device)

                
                bert_priors = {}
                bert_priors['indices'] = bert_priors_data[0].to(args.device)
                bert_priors['values'] = bert_priors_data[1].to(args.device)

                kg_priors = {}
                kg_priors['indices'] = kg_priors_data[0].to(args.device)
                kg_priors['values'] = kg_priors_data[1].to(args.device)

                all_priors = (priors_data, bert_priors, kg_priors)
                # [loss, logits, all_hidden_states, all_attentions]
                outputs = model(data, all_priors)
                loss = outputs[0]
                # print(outputs[1])
                
                if args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                
                tr_loss += loss.detach()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # print(loss.detach())
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1


                if (args.logging_steps > 0 and global_step % args.logging_steps==0):
                    logs = {}
                    tr_loss_scalar = tr_loss.item()
                    logs['loss'] = (tr_loss_scalar - logging_loss_scalar) / args.logging_steps
                    logs['learning_rate'] = scheduler.get_last_lr()[0]
                    logging_loss_scalar = tr_loss_scalar
                    if tb_writer:
                        for k, v in logs.items():
                            if isinstance(v, (int, float)):
                                tb_writer.add_scalar(k, v, global_step)
                        tb_writer.flush()
                
                
                if (args.eval_steps > 0) and (global_step % args.eval_steps==0):
                    logs = {}
                    tr_loss_scalar = tr_loss.item()
                    logs['loss'] = (tr_loss_scalar - logging_loss_scalar) / args.eval_steps
                    logs['learning_rate'] = scheduler.get_last_lr()[0]
                    eval_priors_dataloaders = (eval_priors_dataloader, bert_eval_priors_dataloader, kg_eval_priors_dataloader)
                    metrics, preds = prediction_loop(args, model, eval_dataloader, eval_priors_dataloaders)
                    if not os.path.exists(saved_model_dir):
                        os.mkdir(saved_model_dir)
                    is_save, model_suffixes, best_results = is_save_model(best_results=best_results, metrics=metrics)

                    if is_save:
                        logger.info('**** Checkpoint Eval Results ****')
                        early_stop_steps = 0
                        for key, value in metrics.items():
                            logger.info('{} = {}'.format(key, value))
                            tb_writer.add_scalar(key, value, global_step)
                        output =  f"**logs of {args.task_desc}, steps: {global_step}, loss is {logs['loss']}, lr is {logs['learning_rate']}"
                        print(output)
                        for suffix in model_suffixes:
                            saved_model_to = os.path.join(saved_model_dir, f'{args.task_desc}_{args.fold}_{suffix}.pth')
                            torch.save(model, saved_model_to)
                            logger.info(f"step: {global_step}, save {suffix} model to: {saved_model_to}")
                    else:
                        early_stop_steps += 1
                        for key, value in metrics.items():
                            tb_writer.add_scalar(key, value, global_step)

                # if early_stop_steps >= args.early_stop_time:
                #     break
                # epoch_pbar.update(1) 
                if global_step >= max_steps:
                    break

            if global_step >= max_steps:
                break
        
        train_pbar.close()
        if tb_writer:
            tb_writer.close()
            
        logging.info(f'\n\nTraining completed, global_step: {global_step}')

        torch.save(model, os.path.join(saved_model_dir, f'{args.task_desc}_{args.fold}_final.pth'))
        logger.info(f"save fianl model to: {os.path.join(saved_model_dir, f'{args.task_desc}_{args.fold}_final.pth')}")
    eval_results = {}
    
    if args.do_eval:
        model = torch.load(os.path.join(saved_model_dir, f'{args.task_desc}_{args.fold}_final.pth'))
        model.to(args.device)
        model.eval()
        logger.info('*** Evaluate ***')
        logger.info('Num examples = {}'.format(len(eval_dataloader.dataset)))
        eval_priors_dataloaders = (eval_priors_dataloader, bert_eval_priors_dataloader, kg_eval_priors_dataloader)
        eval_result, preds = prediction_loop(args, model, eval_dataloader, eval_priors_dataloaders)
        output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
        with open(output_eval_file, 'a') as writer:
            writer.write(f"Arguments {args}\n")
            logger.info('*** Eval Results ***')
            for key, value in eval_result.items():
                logger.info("{} = {}\n".format(key, value))
                writer.write('{} = {}\n'.format(key, value))
        eval_results.update(eval_result)

    if args.do_test:
        model = torch.load(os.path.join(saved_model_dir, f'{args.task_desc}_{args.fold}_final.pth'))
        model.to(args.device)
        logging.info('*** Test with final model ***')
        # predict
        test_priors_dataloaders = (test_priors_dataloader, bert_test_priors_dataloader, kg_test_priors_dataloader)
        test_result, preds = prediction_loop(args, model, test_dataloader, test_priors_dataloaders, description='Testing')
        output_test_file = os.path.join(args.output_dir, 'test_results.txt')
        with open(output_test_file, 'a') as writer:
            writer.write(f"Arguments {args}\n")
            logger.info('**** Test with final model ****')
            for key, value in test_result.items():
                logger.info('{} = {}\n'.format(key, value))
                writer.write('{} = {}\n'.format(key, value))
        preds = pd.DataFrame(preds)
        preds.to_csv(os.path.join(log_dir, f'{args.task_desc}_{args.fold}_fianl_results.csv'))
        eval_results.update(test_result)

        if args.label_key == 'expired':
            for suffix in ['overall_best', 'roc_best', 'pr_best', 'loss_best']:
                # predict
                model_path = os.path.join(saved_model_dir, f'{args.task_desc}_{args.fold}_{suffix}.pth')
                best_model = torch.load(model_path)
                best_model.to(args.device)
                best_model.eval()
                test_result, preds = prediction_loop(args, best_model, test_dataloader, test_priors_dataloaders, description='Testing')
                output_test_file = os.path.join(args.output_dir, 'test_results.txt')
                with open(output_test_file, 'a') as writer:
                    logger.info(f'**** Results of {suffix} model****')
                    for key, value in test_result.items():
                        logger.info('{} = {}\n'.format(key, value))
                        writer.write('{} = {}\n'.format(key, value))
                preds = pd.DataFrame(preds)
                preds.to_csv(os.path.join(log_dir, f'{args.task_desc}_{args.fold}_{suffix}_results.csv'))
                eval_results.update(test_result)
        else:
            # predict
            model_path = os.path.join(saved_model_dir, f'{args.task_desc}_{args.fold}_acc_best.pth')
            best_model = torch.load(model_path)
            best_model.to(args.device)
            best_model.eval()
            test_result, preds = prediction_loop(args, best_model, test_dataloader, test_priors_dataloaders, description='Testing')
            output_test_file = os.path.join(args.output_dir, 'test_results.txt')
            with open(output_test_file, 'a') as writer:
                logger.info(f'**** Results of acc_best model****')
                for key, value in test_result.items():
                    logger.info('{} = {}\n'.format(key, value))
                    writer.write('{} = {}\n'.format(key, value))
            preds = pd.DataFrame(preds)
            preds.to_csv(os.path.join(log_dir, f'{args.task_desc}_{args.fold}_acc_best_results.csv'))
            eval_results.update(test_result)



def get_summary(model):
    total_params = 0
    for name, param in model.named_parameters():
        shape = param.shape
        param_size = 1
        for dim in shape:
            param_size *= dim
        print(name, shape, param_size)
        total_params += param_size
    print(total_params)
        

if __name__ == "__main__":
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
    