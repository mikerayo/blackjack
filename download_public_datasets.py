"""
Descargar datasets públicos de blackjack para fine-tuning.

Datasets disponibles:
- 900,000 Hands of Blackjack Results (Kaggle)
- 50 Million Blackjack Hands (Kaggle)
- Simulated Blackjack Data (Kaggle)
"""

import os
import json
import pandas as pd
from pathlib import Path

# Crear directorio para datasets
DATA_DIR = Path("public_datasets")
DATA_DIR.mkdir(exist_ok=True)


def download_kaggle_dataset(dataset_name, files_to_download):
    """
    Descargar dataset de Kaggle usando Kaggle API.

    Requiere:
    1. pip install kaggle
    2. API key de Kaggle en ~/.kaggle/kaggle.json
    """
    try:
        import kaggle
        print(f"📥 Descargando {dataset_name}...")

        for file in files_to_download:
            kaggle.api.dataset_download_file(
                dataset_name,
                file_name=file,
                path=str(DATA_DIR)
            )
            print(f"   ✅ {file} descargado")

        return True
    except ImportError:
        print("❌ Kaggle API no instalada.")
        print("   Instala: pip install kaggle")
        print("   Configura: https://github.com/Kaggle/kaggle-api")
        return False
    except Exception as e:
        print(f"❌ Error descargando {dataset_name}: {e}")
        return False


def process_blackjack_hands(csv_file):
    """Procesar dataset de manos de blackjack."""
    print(f"\n📊 Procesando {csv_file}...")

    try:
        # Leer CSV
        df = pd.read_csv(csv_file)

        print(f"   Total manos: {len(df)}")
        print(f"   Columnas: {list(df.columns)}")

        # Estadísticas básicas
        if 'result' in df.columns:
            print(f"\n   Distribución de resultados:")
            print(df['result'].value_counts())

        # Guardar resumen
        summary = {
            'total_hands': len(df),
            'columns': list(df.columns),
            'shape': df.shape,
            'sample_rows': df.head(5).to_dict('records')
        }

        summary_file = csv_file.parent / f"{csv_file.stem}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"   ✅ Resumen guardado en {summary_file}")

        return df

    except Exception as e:
        print(f"❌ Error procesando {csv_file}: {e}")
        return None


def main():
    """Descargar y procesar datasets públicos."""

    print("="*70)
    print("🎰 DESCARGADOR DE DATASETS PÚBLICOS DE BLACKJACK")
    print("="*70)

    # Lista de datasets a descargar
    datasets = [
        {
            'name': 'mojocolors/900000-hands-of-blackjack-results',
            'files': ['blackjack.csv'],  # Ajustar según nombre real
            'description': '900,000 manos de blackjack'
        },
        {
            'name': 'dennisho/blackjack-hands',
            'files': ['blackjack.csv'],  # Ajustar según nombre real
            'description': '50 millones de manos'
        }
    ]

    # Intentar descargar
    print("\n📥 Intentando descargar datasets de Kaggle...")
    print("⚠️  NOTA: Requiere API key de Kaggle configurada\n")

    downloaded = []
    for dataset in datasets:
        if download_kaggle_dataset(dataset['name'], dataset['files']):
            downloaded.append(dataset)

    # Procesar archivos descargados
    print("\n" + "="*70)
    print("📊 PROCESANDO DATASETS")
    print("="*70)

    # Buscar archivos CSV descargados
    csv_files = list(DATA_DIR.glob("*.csv"))
    print(f"\n✅ Encontrados {len(csv_files)} archivos CSV")

    for csv_file in csv_files:
        process_blackjack_hands(csv_file)

    # Resumen final
    print("\n" + "="*70)
    print("📋 RESUMEN")
    print("="*70)
    print(f"\nDatasets descargados: {len(downloaded)}")
    for ds in downloaded:
        print(f"  - {ds['description']}")

    print(f"\nArchivos disponibles en {DATA_DIR.absolute()}:")
    for file in DATA_DIR.iterdir():
        if file.is_file():
            size_mb = file.stat().st_size / (1024*1024)
            print(f"  - {file.name} ({size_mb:.2f} MB)")

    print("\n💡 Usa estos datasets para fine-tuning:")
    print("   python fine_tune_with_public_data.py --dataset public_datasets/blackjack.csv")


if __name__ == "__main__":
    main()
