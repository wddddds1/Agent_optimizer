Human chr1 WGS alignment benchmark

Reference
- hg38 chr1 from UCSC: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz
- ~249M bp, single chromosome

Reads
- Source: ERR013101 (1000 Genomes, NA12878, Illumina GA-II, 100bp SE)
  https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR013/ERR013101/ERR013101_1.fastq.gz
- Subsampled: first 200,000 reads (800,000 lines)

Typical run
- Index (pre-built, included):
  bwa index chr1.fa
- Align (single thread baseline):
  bwa mem -t 1 chr1.fa reads.fq > aln.sam
  ~41s wall on Apple M3 Pro (16 core), -O2 build
- Align (12 threads):
  bwa mem -t 12 chr1.fa reads.fq > aln.sam
  ~10s wall
