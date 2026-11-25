import math

from cs336_scaling.modal_utils import MODAL_SECRETS, VOLUME_MOUNTS, app, build_image
from cs336_scaling.tokenized_data import (
    DCLMData,
    ShuffledDCLMShardUrls,
    TokenizedDCLMChunk,
    TokenizedDCLMChunkResult,
)


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    secrets=MODAL_SECRETS,
    # max_containers=500,# should use this when making the full data
    max_containers=100,
    timeout=60 * 60 * 12,
)
def make_chunk(dclm_chunk: TokenizedDCLMChunk) -> TokenizedDCLMChunkResult:
    try:
        if dclm_chunk.is_completed():
            print("Existing result", res := dclm_chunk.try_load())
            return res
    except Exception as e:
        print(e)
    if dclm_chunk.data_dir.exists():
        import shutil

        print("Deleting the old version")
        shutil.rmtree(dclm_chunk.data_dir)
    print("New result", res := dclm_chunk.load_or_create(use_lock=False))
    return res


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    timeout=60 * 60 * 12,
    # Raw chunks + shuffle runs + final shards each hold ~2 bytes/token locally.
    # 512 GiB is Modal's minimum explicit request.
    ephemeral_disk=512 * 1024,
    memory=32 * 1024,  # one shuffle run (DCLMData.shuffle_run_target_bytes) in RAM
    nonpreemptible=True,
)
def make_final_dclm_data():
    dclm_data = DCLMData()
    print(dclm_data.data_dir)
    print(dclm_data.load_or_create())


@app.local_entrypoint()
def modal_main():
    # tokenize chunks in parallel. can move this into the make final dclm data, but this is nicer. approx 150M tokens per chunk (for settings n_dclm_chunks)
    dclm_shard_urls = ShuffledDCLMShardUrls().load_or_create()
    # Enough ~150M-token chunks to cover DCLMData's train + validation tokens,
    # with 25% margin since chunk sizes vary. DCLMData asserts if it falls short.
    dclm_data = DCLMData()
    padded_train_tokens = (
            math.ceil(dclm_data.total_tokens / dclm_data.tokens_per_chunk)
            * dclm_data.tokens_per_chunk
    )
    needed_tokens = padded_train_tokens + dclm_data.validation_tokens
    n_dclm_chunks = math.ceil(needed_tokens / 150_000_000 * 1.25)
    print(f"tokenizing {n_dclm_chunks} DCLM chunks for {needed_tokens:_} tokens")
    total_tokens = 0
    chunk_idx = 0
    for chunk in make_chunk.map(
        [TokenizedDCLMChunk(url=dclm_shard_urls[i]) for i in range(n_dclm_chunks)]
    ):
        total_tokens += chunk.n_tokens
        chunk_idx += 1
        print(f"{chunk_idx=} {total_tokens=:_}")

    # make final data
    make_final_dclm_data.remote()
