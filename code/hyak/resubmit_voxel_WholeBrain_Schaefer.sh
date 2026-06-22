for i in {180..185}; do
  STAGE11_CHUNKS=500 STAGE11_CHUNK_IDX=$i bash hyak/submit_mvpa_L2_schaefer_stage.sh 11:SAD
done
#Once Done
STAGE11_CHUNKS=500 bash hyak/submit_mvpa_L2_schaefer_stage.sh 11:SAD:merge

