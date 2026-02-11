E. coli K-12 (real data, small/medium)

Reference
- Source (NCBI): GCF_000005845.2_ASM584v2
  https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz
- Local file: ref.fa (decompressed)

Reads
- Source (ENA): SRR2584866_1.fastq.gz
  https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR258/006/SRR2584866/SRR2584866_1.fastq.gz
- We subsampled the first 200,000 reads (800,000 lines) to keep size reasonable:
  curl -L <URL> | gzip -dc | head -n 800000 > reads.fq

Typical run
- Index:
  bwa index ref.fa
- Align (12 threads):
  bwa mem -t 12 ref.fa reads.fq > aln.sam

Larger read set
- reads_2m.fq: first 2,000,000 reads (8,000,000 lines) from SRR2584866_1.fastq.gz
