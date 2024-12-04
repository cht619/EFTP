# EFTP
Implementation of Effective Test-Time Personalization for federated Semantic Segmentation.

### Federated Training



⬇️ Refer [FedSeg](https://github.com/lightas/FedSeg) to train to global model. 

For instance, in the ACDC dataset (Fog, Night, Rain, Snow), if Snow is designated as the test domain, 
the remaining three domains (Fog, Night, Rain) are used as the training domains.

## Test-time Personalization
```
bash TTA/exp/train_acdc.sh 0 Snow EFTP 0
```

The key codes of EFTP:

```python
class EFTP:
    def __init__(self,):
        self.model_global = create_global_model(self.model)  # copy model
        self.upscaling_layer()  # upscaling linear to WMoE
        self.model_initialization()  # initialize local model
        
        # initialize: prompt and router
    def model_initialization(self):
        local_model = deepcopy(self.model)

        ema_model_params = []
        for n, p in self.model_global.named_parameters():
            if any(s in n for s in ['prompt', 'gate', 'attn.proj']):  # update
                p.requires_grad = True
                ema_model_params.append(p)
            else:
                p.requires_grad = False

        local_model_params = []
        for n, p in local_model.named_parameters():
            local_model_params.append(p)
            p.requires_grad = True
        all_params_list = [
            {'params': ema_model_params, 'lr': 1e-4},  # local model prompt
            {'params': local_model_params, 'lr': 1e-4},  # expert gate
        ]

        optimizer = optim.AdamW(all_params_list, lr=1e-5, weight_decay=0.01)
        self.model_global.eval()
        train_dataloader = self.get_dataloader()
        for data in tqdm(train_dataloader):
            data = self.process_batch_class_label_nthu(data)
            optimizer.zero_grad()
            batch_size = len(data['inputs'])
            # print(data['data_samples'][0].img_path, data['data_samples'][0].seg_map_path)
            gts = torch.cat([data['data_samples'][i].gt_sem_seg.data.long() for i in range(batch_size)]).cuda()
            outputs = self.model_global.test_step(data)
            seg_logits = torch.cat([outputs[i].seg_logits.data.unsqueeze(0) for i in range(batch_size)]).cuda()
            loss_train = self.model_global.decode_head.loss_decode(seg_logits, gts, ignore_index=255)
            if loss_train < 1.2:  # threshold \tau
                loss_train.backward()
                optimizer.step()

                for name, module in self.model_global.backbone.named_modules():
                    if isinstance(module, WMoE):
                        if name in self.upscaling_names:
                            linear_expert = deepcopy(get_named_submodule(local_model.backbone, name))
                            module.update_expert_model(linear_expert.state_dict())

    # consistency loss
    def forward_loss(self, data, optimizer):
        outputs_logits_ema = self.model_global.test_step(data)[0].seg_logits.data.unsqueeze(0)
        outputs_logits = self.model.test_step(data)[0].seg_logits.data.unsqueeze(0)

        prob_a = F.log_softmax(outputs_logits.reshape(-1, 19), dim=1)
        prob_b = F.softmax(outputs_logits_ema.reshape(-1, 19), dim=1)

        kl_loss = F.kl_div(prob_a, prob_b, reduction='batchmean')
        kl_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # test-time model merging
    def model_merging(self, data):
        for name, module in self.model_global.backbone.named_modules():
            if isinstance(module, WMoE):
                if name in self.upscaling_names:
                    linear_expert = deepcopy(get_named_submodule(self.model.backbone, name))
                    module.update_expert_model(linear_expert.state_dict())

        with torch.no_grad():
            outputs = self.model_global.test_step(data)

        return outputs

    # get result
    def forward_epoch(self, epoch, data, optimizer=None):
        optimizer.zero_grad()
        self.forward_loss(data, optimizer)  # update local model
        outputs = self.model_merging(data)
        return outputs

```
