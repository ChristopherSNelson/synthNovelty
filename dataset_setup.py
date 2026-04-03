import os
import pandas as pd
from datasets import load_dataset
import re

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Mapping of first patent number issued each year (Utility)
# Source: https://www.uspto.gov/web/offices/ac/ido/oeip/taf/issuyear.htm
YEAR_MAP = {
    1976: 3930271,
    1977: 4000521,
    1978: 4065812,
    1979: 4131952,
    1980: 4182000,
    1981: 4242757,
    1982: 4308622,
    1983: 4366578,
    1984: 4424592,
    1985: 4490855,
    1986: 4562596,
    1987: 4633529,
    1988: 4716597,
    1989: 4796306,
    1990: 4891845,
    1991: 5000000,
    1992: 5077836,
    1993: 5175887,
    1994: 5274845,
    1995: 5377361,
    1996: 5479659,
    1997: 5590419,
    1998: 5704064,
    1999: 5854992,
    2000: 6010000,
    2001: 6167568,
    2002: 6330720,
    2003: 6502242,
    2004: 6671882,
    2005: 6836898,
    2006: 6981280,
    2007: 7155743,
    2008: 7313824,
    2009: 7472428,
    2010: 7640601,
    2011: 7861317,
    2012: 8087095,
    2013: 8341762,
    2014: 8621664,
    2015: 8925116,
    2016: 9226429,
}

def get_year_from_id(patent_id):
    if not isinstance(patent_id, str):
        return None
    # Extract numeric part
    # Format: US08188092B2 or US05849732
    match = re.search(r'US(\d+)', patent_id)
    if not match:
        # Check for year-based IDs US2015...
        if patent_id.startswith('US20'):
            try:
                return int(patent_id[2:6])
            except:
                pass
        return None
    
    num_str = match.group(1)
    # Remove leading zeros but keep enough to be > 1M
    num = int(num_str)
    
    # Simple binary search / iterative check
    found_year = 1976
    for year in sorted(YEAR_MAP.keys()):
        if num >= YEAR_MAP[year]:
            found_year = year
        else:
            break
    return found_year

def download_and_process():
    print("Downloading USPTO-50K (pingzhili)...")
    ds = load_dataset("pingzhili/uspto-50k")
    print(ds)
    
    # Combine splits for time-based re-splitting
    full_df = pd.concat([pd.DataFrame(ds[split]) for split in ds])
    print(f"Total rows: {len(full_df)}")
    
    # Drop rows without rxn_smiles
    full_df = full_df.dropna(subset=['rxn_smiles'])
    
    # Map years
    full_df['year'] = full_df['id'].apply(get_year_from_id)
    
    # Check coverage
    missing_years = full_df['year'].isna().sum()
    print(f"Missing years: {missing_years}")
    
    # Remove rows with missing years
    full_df = full_df.dropna(subset=['year'])
    
    # Save the combined dataset first
    full_df.to_csv(f"{DATA_DIR}/uspto_all.csv", index=False)
    
    # Define Time Split: Everything before 2013 is Train/Val, 2013+ is Test
    # Or 2014+ is Test to be more "novel".
    # Let's check distribution
    print("\nYear distribution:")
    print(full_df['year'].value_counts().sort_index())
    
    cutoff = 2016
    train_val_df = full_df[full_df['year'] < cutoff].copy()
    test_df = full_df[full_df['year'] >= cutoff].copy()
    
    print(f"\nSplit at year {cutoff}:")
    print(f"Train/Val: {len(train_val_df)}")
    print(f"Test (Post-{cutoff}): {len(test_df)}")
    
    # Further split Train/Val into Train and Val (randomly, within the pre-cutoff set)
    train_val_df = train_val_df.sample(frac=1, random_state=42) # Shuffle
    val_size = int(len(train_val_df) * 0.1) # 10% for validation
    
    val_df = train_val_df.iloc[:val_size]
    train_df = train_val_df.iloc[val_size:]
    
    # Save splits
    train_df.to_csv(f"{DATA_DIR}/uspto_train.csv", index=False)
    val_df.to_csv(f"{DATA_DIR}/uspto_val.csv", index=False)
    test_df.to_csv(f"{DATA_DIR}/uspto_test.csv", index=False)
    
    print(f"Saved: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")

if __name__ == "__main__":
    download_and_process()
