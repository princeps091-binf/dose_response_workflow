import pandas as pd
from pathlib import Path
import gzip

def import_vcf_as_df(vcf_file):
    with gzip.open(vcf_file, 'rt') as f:
        header_lines = sum(1 for line in f if line.startswith('##'))
    # Read into a DataFrame, skipping the header lines
    df = pd.read_csv(vcf_file, sep='\t', skiprows=header_lines)
    return df

def extract_gene_anno(anno_string):
    tmp_anno_list = [s for s in anno_string.split(';') if "VW=" in s]
    return [s.split('|')[0][3:] for s in tmp_anno_list]

def extract_variant_class(anno_string):
    tmp_anno_list = [s for s in anno_string.split(';') if "VC=" in s]
    return [s.split('=')[1] for s in tmp_anno_list]
def extract_variant_type(anno_string):
    tmp_anno_list = [s for s in anno_string.split(';') if "VT=" in s]
    return [s.split('=')[1] for s in tmp_anno_list]

def get_vcf_summary_tbl(vcf_file):
    with gzip.open(vcf_file, 'rt') as f:
        header_lines = sum(1 for line in f if line.startswith('##'))
    df = pd.read_csv(vcf_file, sep='\t', skiprows=header_lines)
    anno_gene_symbol = [extract_gene_anno(anno) for anno in df.INFO.to_list()]
    variant_class_list = [extract_variant_type(anno) for anno in df.INFO.to_list()]
    variant_type_list = [extract_variant_class(anno) for anno in df.INFO.to_list()]

    return df.loc[:,['#CHROM','POS']].rename(columns={'#CHROM':'chrom'}).assign(gene = anno_gene_symbol,var_class = variant_class_list,var_type = variant_type_list, sanger_model_id = vcf_file.stem.split('_')[0]).explode('gene').explode('var_class').explode('var_type')

def parse_gmt(file_path):
    gene_sets = {}
    with open(file_path, 'r') as f:
        for line in f:
            # GMT files use tabs to separate columns
            columns = line.strip().split('\t')
            
            # Index 0: Set Name
            # Index 1: Description (usually ignored in analysis)
            # Index 2 onwards: The Genes
            set_name = columns[0]
            genes = [g for g in columns[2:] if g]  # Filter out empty strings
            
            gene_sets[set_name] = genes
            
    return gene_sets


