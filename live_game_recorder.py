"""
Monitor en tiempo real mientras juegas en un casino online.

Este script NO interactúa con el casino.
Solo te pide los datos de cada mano y los guarda.

Es más rápido que el bot interactivo.
"""

import json
import sys
from datetime import datetime
from pathlib import Path


class LiveGameRecorder:
    """Grabador rápido de manos en tiempo real."""

    def __init__(self, save_dir='casino_data'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.session = {
            'casino': '',
            'start_time': None,
            'hands': []
        }

        # Atajos para escribir más rápido
        self.shortcuts = {
            'h': 'hit',
            's': 'stand',
            'd': 'double',
            'sp': 'split',
            'su': 'surrender',
            'w': 'win',
            'l': 'loss',
            'p': 'push',
            'bj': 'blackjack'
        }

    def start(self, casino_name):
        """Iniciar sesión."""
        self.session['casino'] = casino_name
        self.session['start_time'] = datetime.now().isoformat()

        print(f"\n{'='*60}")
        print(f"🎰 GRABANDO: {casino_name}")
        print(f"{'='*60}")
        print(f"Inicio: {datetime.now().strftime('%H:%M:%S')}")
        print(f"\nAtajos:")
        print(f"  Acciones: h=hit, s=stand, d=double, sp=split, su=surrender")
        print(f"  Resultados: w=win, l=loss, p=push, bj=blackjack")
        print(f"\nFormato: <tu_total> <dealer_card> <accion> <resultado>")
        print(f"Ejemplo: 18 7 s w")
        print(f"Para terminar: Enter (sin datos)")
        print(f"{'='*60}\n")

    def record_hand(self, player_total, dealer_card, action, result, bet=10):
        """Registrar una mano."""
        # Expandir atajos
        action = self.shortcuts.get(action.lower(), action)
        result = self.shortcuts.get(result.lower(), result)

        hand = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'player_total': int(player_total),
            'dealer_card': int(dealer_card),
            'action': action,
            'result': result,
            'bet': float(bet)
        }

        self.session['hands'].append(hand)

        # Estadísticas rápidas
        total = len(self.session['hands'])
        wins = sum(1 for h in self.session['hands'] if h['result'] in ['win', 'blackjack'])

        print(f"✅ Mano #{total}: {player_total} vs {dealer_card} → {action} → {result} | Win rate: {wins/total:.1%}")

    def save(self):
        """Guardar sesión."""
        if not self.session['hands']:
            print("⚠️  No hay manos para guardar.")
            return

        # Estadísticas
        hands = self.session['hands']
        total = len(hands)
        wins = sum(1 for h in hands if h['result'] in ['win', 'blackjack'])
        losses = sum(1 for h in hands if h['result'] == 'loss')
        pushes = sum(1 for h in hands if h['result'] == 'push')
        bjs = sum(1 for h in hands if h['result'] == 'blackjack')

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.save_dir / f"live_{session_id}.json"

        summary = {
            'session_id': session_id,
            'casino': self.session['casino'],
            'start_time': self.session['start_time'],
            'end_time': datetime.now().isoformat(),
            'stats': {
                'total_hands': total,
                'wins': wins,
                'losses': losses,
                'pushes': pushes,
                'blackjacks': bjs,
                'win_rate': wins/total if total > 0 else 0
            },
            'hands': hands
        }

        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"💾 GUARDADO: {filename}")
        print(f"Manos: {total} | Wins: {wins} ({wins/total:.1%})")
        print(f"{'='*60}\n")

        return filename


def main():
    """Ejecutar grabador."""

    print("""
╔════════════════════════════════════════════════════════════╗
║   🎮 GRABADOR EN TIEMPO REAL                             ║
║   Juega en tu casino mientras este script registra       ║
╚════════════════════════════════════════════════════════════╝
    """)

    recorder = LiveGameRecorder()

    # Configurar
    casino = input("🎰 Nombre del casino: ").strip() or "Unknown"
    recorder.start(casino)

    # Loop principal
    print("\n🎮 Jugando...\n")

    hand_num = 1
    while True:
        try:
            # Prompt corto para entrar rápido
            prompt = f"Mano #{hand_num} <total dealer acción resultado> (Enter para terminar): "
            user_input = input(prompt).strip()

            if not user_input:
                break

            # Parsear entrada
            parts = user_input.split()

            if len(parts) < 4:
                print("❌ Formato incorrecto. Ejemplo: 18 7 s w")
                continue

            player_total = parts[0]
            dealer_card = parts[1]
            action = parts[2]
            result = parts[3]
            bet = parts[4] if len(parts) > 4 else 10

            # Validar números
            try:
                player_total = int(player_total)
                dealer_card = int(dealer_card)
                bet = float(bet)
            except ValueError:
                print("❌ Total y carta dealer deben ser números")
                continue

            # Registrar
            recorder.record_hand(player_total, dealer_card, action, result, bet)
            hand_num += 1

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrumpido")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    # Guardar
    recorder.save()

    # Preguntar si sigue
    continuar = input("¿Empezar nueva sesión? [y/N]: ").lower()
    if continuar == 'y':
        main()  # Recursivo para nueva sesión
    else:
        print("\n✅ Todas las sesiones guardadas en casino_data/")
        print("📊 Para ver todas las sesiones:")
        print("   python -c \"import json; print(json.load(open('casino_data/live_*.json')[0]))\"")


if __name__ == "__main__":
    main()
