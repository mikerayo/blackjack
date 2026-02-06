"""Script para buscar las mejores ofertas en vast.ai"""

import subprocess
import json
import sys

def search_vast_offers():
    """Busca las mejores GPUs en vast.ai para nuestro caso de uso."""

    print("="*80)
    print("BUSCANDO MEJORES OFERTAS EN VAST.AI")
    print("="*80)

    # Parámetros de búsqueda optimizados
    search_cmd = [
        "vast", "search", "offers",
        "--GPU", "RTX_3090,RTX_4090,A100,A40",  # GPUs recomendadas
        "--min-ram", "16",  # Mínimo 16 GB RAM
        "--min-disk", "50",  # Mínimo 50 GB disco
        "--internet", "all",  # Requiere internet para descargar dependencias
        "--raw",  # Output JSON
        "--limit", "20"  # Top 20 ofertas
    ]

    print("\n🔍 Buscando GPUs...")
    print("   GPUs objetivo: RTX 3090, RTX 4090, A100, A40")
    print("   RAM mínima: 16 GB")
    print("   Disco mínimo: 50 GB")
    print("   Requiere: Internet\n")

    try:
        result = subprocess.run(search_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Error en búsqueda: {result.stderr}")
            print("\n💡 Solución:")
            print("   1. Instalar vast.ai: pip install vast-ai")
            print("   2. Iniciar sesión: vast.ai login")
            return None

        # Parsear resultados JSON
        offers = json.loads(result.stdout)

        if not offers:
            print("❌ No se encontraron ofertas disponibles")
            return None

        print(f"✅ Encontradas {len(offers)} ofertas\n")
        print("="*80)
        print("TOP 10 MEJORES OFERTAS")
       ("="*80)

        # Mostrar top 10
        for i, offer in enumerate(offers[:10], 1):
            print(f"\n📦 Oferta #{i}:")
            print(f"   GPU: {offer.get('gpu_name', 'Unknown')}")
            print(f"   VRAM: {offer.get('gpu_ram', 'Unknown')} GB")
            print(f"   RAM: {offer.get('cpu_ram', 'Unknown')} GB")
            print(f"   Precio: ${offer.get('dph_total', 0):.4f}/hora")
            print(f"   Precio estimado 3h: ${offer.get('dph_total', 0) * 3:.2f}")
            print(f"   SSD: {offer.get('disk_space', 'Unknown')} GB")
            print(f"   ID: {offer.get('id', 'Unknown')}")
            print(f"   Reputation: {offer.get('reliability2', 'Unknown')}")

            if i == 1:
                best_id = offer.get('id')
                best_gpu = offer.get('gpu_name')

        # Guardar la mejor oferta
        if offers:
            best = offers[0]
            print("\n" + "="*80)
            print("💡 MEJOR OFERTA RECOMENDADA:")
            print("="*80)
            print(f"   GPU: {best.get('gpu_name')}")
            print(f"   Precio: ${best.get('dph_total', 0):.4f}/hora")
            print(f"   Costo estimado 3h: ${best.get('dph_total', 0) * 3:.2f}")
            print(f"   ID: {best.get('id')}")

            print("\n🚀 Para alquilar esta GPU:")
            print(f"   vast create {best.get('id')}")

        return offers

    except json.JSONDecodeError:
        print("❌ Error al parsear resultados de vast.ai")
        print("\n💡 Solución alternativa:")
        print("   Visita https://vast.ai/create")
        print("   Filtra por: RTX 3090, RTX 4090, o A100")
        print("   Ordena por precio (DPH)")
        return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def create_instance(offer_id):
    """Crea una instancia con la oferta especificada."""

    print(f"\n🚀 Creando instancia con oferta {offer_id}...")

    create_cmd = [
        "vast", "create", "instance", str(offer_id),
        "--image", "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
        "--disk", "50",  # 50 GB
        "--raw"
    ]

    result = subprocess.run(create_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        instance_info = json.loads(result.stdout)
        print("✅ Instancia creada exitosamente!")
        print(f"\n📋 Detalles:")
        print(f"   Instance ID: {instance_info.get('new_contract')}")
        print(f"   SSH: {instance_info.get('ssh')}")
        print("\n💡 Comandos útiles:")
        print(f"   vast ssh {instance_info.get('new_contract')}")
        print(f"   vast destroy {instance_info.get('new_contract')}")
        return instance_info
    else:
        print(f"❌ Error al crear instancia: {result.stderr}")
        return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Buscar y crear instancias en vast.ai')
    parser.add_argument('--search', action='store_true', help='Buscar ofertas')
    parser.add_argument('--create', type=str, help='Crear instancia con ID de oferta')

    args = parser.parse_args()

    if args.search:
        search_vast_offers()
    elif args.create:
        create_instance(args.create)
    else:
        # Por defecto, buscar
        search_vast_offers()
