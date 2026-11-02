from safetensors.torch import load_file
"""BERT + four lexical projections + dynamic feature fusion + seven logits."""
import runtime
import torch
from torch import nn
from transformers import BertConfig, BertModel, AutoTokenizer
from features import FEATURE_NAMES, LABELS

ARCHITECTURE = 'primary7-grouped-v3-scope-v4-context-disgust-caring'


class PrimaryClassifier(nn.Module):
    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        def project(a, b):
            return nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout))
        self.task = project(24, 8)
        self.source = project(8, 4)
        self.expression = project(100, 32)
        self.modifier = project(12, 4)
        self.feature_fusion = project(len(FEATURE_NAMES) - 96, 64)
        self.text_norm = nn.LayerNorm(encoder.config.hidden_size)
        self.classifier = nn.Sequential(nn.Linear(encoder.config.hidden_size+64, 256),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 7))

    def forward(self, numeric, input_ids, attention_mask, token_type_ids):
        if numeric.ndim != 2 or numeric.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f'Expected normalized features [B,{len(FEATURE_NAMES)}]')
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids).last_hidden_state[:, 0]
        f = numeric
        grouped = torch.cat([self.task(f[:, :24]), self.source(f[:, 24:32]),
                             self.expression(f[:, 32:132]), self.modifier(f[:, 132:144]),
                             f[:, 144:167], f[:, 167:197], f[:, 197:211], f[:, 211:]], dim=-1)
        u = self.feature_fusion(grouped)
        return self.classifier(torch.cat([self.text_norm(cls), u], dim=-1))


def new_model(cfg):
    path = runtime.resolve(cfg['pretrained_path'])
    encoder = BertModel.from_pretrained(str(path), local_files_only=True, add_pooling_layer=False)
    if cfg['max_length'] > encoder.config.max_position_embeddings:
        raise ValueError('Input length exceeds BERT position capacity')
    return PrimaryClassifier(encoder, cfg['dropout'])


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
                'label_order': LABELS, 'feature_code_sha256': runtime.sha256(runtime.BASE / 'features.py')}
    runtime.write_json(folder / 'metadata.json', metadata)


def load_checkpoint(folder, allow_check=False):
    folder = runtime.resolve(folder)
    metadata = runtime.read_json(folder / 'metadata.json')
    if metadata.get('check_only') and not allow_check:
        raise ValueError('This checkpoint is an integrity-check fixture, not a trained model')
    if metadata.get('architecture') != ARCHITECTURE or metadata.get('feature_names') != FEATURE_NAMES or metadata.get('label_order') != LABELS:
        raise ValueError('Checkpoint architecture/feature/label order mismatch')
    if metadata.get('feature_code_sha256') != runtime.sha256(runtime.BASE / 'features.py'):
        raise ValueError('Feature/rule implementation changed since training')
    config = BertConfig.from_pretrained(str(folder / 'encoder_config'), local_files_only=True)
    model = PrimaryClassifier(BertModel(config, add_pooling_layer=False), metadata['model']['dropout'])
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
