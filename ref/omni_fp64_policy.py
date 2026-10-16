"""Fixed contract for the separately authorized FP64 reference lane."""
import hashlib
import importlib
from pathlib import Path

REVISION = '0db1cd2607d76a7bdb2a382f659e7b313079f84b'
CORPORA = {'text': 'corpus', 'omni-vision': 'corpus_omni_vision',
           'omni-video': 'corpus_omni_video', 'audio': 'corpus_audio',
           'omni-av': 'corpus_omni_av', 'omni-long': 'corpus_omni_long'}
POLICY = dict(
    name='independent-cpu-fp64-v1',
    weights='model-card-bfloat16-then-float64',
    preprocessing='pinned-processor-float32',
    inference='cpu-float64-decoder-media-towers-rotary-head',
    input='pinned-request-encoding-only',
    routing='independent-float64-softmax-topk-diagnostic-only',
    query_block=128, key_block=2048, first_layer=0,
    dtype_audit='every-floating-operation',
    layer_relative_l2=.01, max_logit_error=.05, max_probability_error=.002,
    exact_tokens=True, exact_argmax=True,
)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def implementation_sources():
    ref = Path(__file__).resolve().parent
    return {str(path.relative_to(ref.parent)): digest(path) for path in
            [ref/name for name in ('omni_fp64.py', 'oracle_fp64.py', 'omni_fp64_policy.py',
                                  'omni_weights.py', 'omni_media.py', 'golden_io.py')]}


def runtime_provenance(model_dir, corpus):
    import torch
    import transformers
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe
    from transformers import vision_utils
    root = Path(__file__).resolve().parents[1]
    model_dir = Path(model_dir).resolve()
    module = importlib.import_module(CORPORA[corpus])
    paths = [Path(module.__file__), Path(modeling_qwen3_omni_moe.__file__), Path(vision_utils.__file__),
             model_dir/'joint_schema_model.py', model_dir/'config.json', model_dir/'joint_head_config.json',
             root/'tools/verify_snapshot.py', root/'tools/snapshots'/(REVISION+'.json')]
    return dict(snapshot_path=str(model_dir), torch=torch.__version__, transformers=transformers.__version__,
                source_sha256={str(p.resolve()):digest(p) for p in paths})


def validate_policy(manifest):
    if (manifest.get('complete') is not True or manifest.get('revision') != REVISION or
            manifest.get('precision') != 'float64-streamed' or
            manifest.get('independent_from_native') is not True or 'arbitration' in manifest or
            manifest.get('fp64_policy') != POLICY or
            manifest.get('implementation_sha256') != implementation_sources()):
        raise ValueError('FP64 qualification requires the complete independent CPU FP64 policy and current sources')
    if (manifest.get('corpus') not in CORPORA or not isinstance(manifest.get('snapshot_path'),str) or
            not Path(manifest['snapshot_path']).is_absolute()):
        raise ValueError('missing FP64 runtime provenance')
    expected = runtime_provenance(manifest['snapshot_path'], manifest['corpus'])
    if any(manifest.get(key) != value for key,value in expected.items()):
        raise ValueError('FP64 runtime provenance differs from source files or library versions')


def verify_artifacts(path, manifest):
    expected = {'requests.jsonl', 'encoded.jsonl', 'logits.safetensors'} | {
        f'layers/{record["id"]}.safetensors' for record in manifest['records']}
    hashes = manifest.get('artifacts_sha256', {})
    if set(hashes) != expected:
        raise ValueError('FP64 artifact inventory differs')
    for name in sorted(expected):
        target = (Path(path)/name).resolve()
        if not target.is_relative_to(Path(path).resolve()) or digest(target) != hashes[name]:
            raise ValueError(f'FP64 artifact hash mismatch: {name}')
