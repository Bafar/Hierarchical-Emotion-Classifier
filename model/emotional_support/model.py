from safetensors.torch import load_file
"""BERT + grouped 95 text features + five conditional logits."""
import runtime
import torch
from torch import nn
from transformers import BertConfig, BertModel, AutoTokenizer
from features import FEATURE_NAMES, LABELS

ARCHITECTURE = 'secondary-emotional-grouped-v3-rejection-scope-v4-context-disgust-caring'


class SecondaryClassifier(nn.Module):
    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        def project(a, b):
            return nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout))
        self.context = project(12, 4)
        self.stance = project(40, 20)
        self.modifier = project(8, 4)
        self.feature_fusion = project(len(FEATURE_NAMES) - 32, 32)
        self.text_norm = nn.LayerNorm(encoder.config.hidden_size)
        self.classifier = nn.Sequential(nn.Linear(encoder.config.hidden_size+32, 128),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(128, len(LABELS)))

    def forward(self, numeric, input_ids, attention_mask, token_type_ids):
        if numeric.ndim != 2 or numeric.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f'Expected normalized features [B,{len(FEATURE_NAMES)}]')
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids).last_hidden_state[:, 0]
        f = numeric
        grouped = torch.cat([self.context(f[:, :12]), self.stance(f[:, 12:52]),
                             self.modifier(f[:, 52:60]),
                             f[:, 60:72], f[:, 72:83], f[:, 83:95], f[:, 95:]], dim=-1)
        u = self.feature_fusion(grouped)
        return self.classifier(torch.cat([self.text_norm(cls), u], dim=-1))


def new_model(cfg):
    path = runtime.resolve(cfg['pretrained_path'])
    encoder = BertModel.from_pretrained(str(path), local_files_only=True, add_pooling_layer=False)
    freeze_bottom = cfg.get('freeze_bottom_layers', 2)
    if not isinstance(freeze_bottom, int) or not 0 <= freeze_bottom <= encoder.config.num_hidden_layers:
        raise ValueError('freeze_bottom_layers out of range')
    for layer in encoder.encoder.layer[:freeze_bottom]:
        for p in layer.parameters():
            p.requires_grad = False
    if cfg['max_length'] > encoder.config.max_position_embeddings:
        raise ValueError('Input length exceeds BERT position capacity')
    return SecondaryClassifier(encoder, cfg['dropout'])


def tokenizer_from(path):
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    tok.padding_side = 'right'
    tok.truncation_side = 'right'
    if any(x is None for x in (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)):
        raise ValueError('BERT tokenizer missing CLS/SEP/PAD')
    return tok


def save_checkpoint(folder, model, tokenizer, metadata):
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / 'model.pt')
    model.encoder.config.save_pretrained(folder / 'encoder_config')
    tokenizer.save_pretrained(folder / 'tokenizer')
    metadata = {**metadata, 'architecture': ARCHITECTURE, 'feature_names': FEATURE_NAMES,
                'label_order': LABELS, 'feature_code_sha256': runtime.sha256(runtime.BASE / 'features.py'),
                'primary_feature_sha256': runtime.sha256(runtime.ROOT / 'primary_emotion_classifier/features.py')}
    runtime.write_json(folder / 'metadata.json', metadata)


def load_checkpoint(folder, allow_check=False):
    folder = runtime.resolve(folder)
    metadata = runtime.read_json(folder / 'metadata.json')
    if metadata.get('check_only') and not allow_check:
        raise ValueError('This checkpoint is an integrity-check fixture, not a trained model')
    if metadata.get('architecture') != ARCHITECTURE or metadata.get('feature_names') != FEATURE_NAMES or metadata.get('label_order') != LABELS:
        raise ValueError('Checkpoint architecture/feature/label order mismatch')
    if metadata.get('feature_code_sha256') != runtime.sha256(runtime.BASE / 'features.py') or metadata.get('primary_feature_sha256') != runtime.sha256(runtime.ROOT / 'primary_emotion_classifier/features.py'):
        raise ValueError('Feature/rule implementation changed since training')
    config = BertConfig.from_pretrained(str(folder / 'encoder_config'), local_files_only=True)
    encoder = BertModel(config, add_pooling_layer=False)
    for layer in encoder.encoder.layer[:metadata['model']['freeze_bottom_layers']]:
        for p in layer.parameters():
            p.requires_grad = False
    model = SecondaryClassifier(encoder, metadata['model']['dropout'])
    model.load_state_dict(load_file(str(folder / 'model.safetensors'), device='cpu'), strict=True)
    return model, tokenizer_from(folder / 'tokenizer'), metadata


def optimizer_groups(model, cfg):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        encoder = name.startswith('encoder.')
        decay = p.ndim > 1 and not name.endswith('.bias')
        key = (encoder, decay)
        groups.setdefault(key, []).append(p)
    return [{'params': ps, 'lr': cfg['encoder_learning_rate'] if enc else cfg['head_learning_rate'],
             'weight_decay': cfg['weight_decay'] if decay else 0.}
            for (enc, decay), ps in groups.items()]
