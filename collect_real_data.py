"""
Script para recopilar datos REALES de casinos mientras juegas.
Usa estos datos para fine-tuning del modelo.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
import numpy as np

class CasinoDataCollector:
    """Recopila datos de partidas reales de casino."""

    def __init__(self, save_dir='casino_data'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.current_session = {
            'session_id': None,
            'start_time': None,
            'hands': [],
            'casino_name': '',
            'table_rules': {},
        }

    def start_session(self, casino_name="Unknown", **rules):
        """Iniciar nueva sesión de juego."""
        self.current_session.update({
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'start_time': datetime.now().isoformat(),
            'casino_name': casino_name,
            'table_rules': rules,
            'hands': []
        })
        print(f"✅ Sesión iniciada: {self.current_session['session_id']}")

    def record_hand(self, state, action, reward, result, bet_amount=10, **metadata):
        """Registrar una mano jugada."""
        hand_data = {
            'timestamp': datetime.now().isoformat(),
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': int(action),
            'reward': float(reward),
            'result': result,  # 'win', 'loss', 'push', 'blackjack'
            'bet_amount': float(bet_amount),
            **metadata
        }
        self.current_session['hands'].append(hand_data)

    def end_session(self):
        """Terminar sesión y guardar datos."""
        if not self.current_session['hands']:
            print("⚠️  No hay manos para guardar.")
            return

        # Calcular estadísticas
        hands = self.current_session['hands']
        total_hands = len(hands)
        wins = sum(1 for h in hands if h['result'] in ['win', 'blackjack'])
        losses = sum(1 for h in hands if h['result'] == 'loss')
        pushes = sum(1 for h in hands if h['result'] == 'push')
        total_profit = sum(h['reward'] for h in hands)

        summary = {
            'session_id': self.current_session['session_id'],
            'start_time': self.current_session['start_time'],
            'end_time': datetime.now().isoformat(),
            'casino_name': self.current_session['casino_name'],
            'table_rules': self.current_session['table_rules'],
            'statistics': {
                'total_hands': total_hands,
                'wins': wins,
                'losses': losses,
                'pushes': pushes,
                'win_rate': wins / total_hands if total_hands > 0 else 0,
                'total_profit': total_profit,
                'avg_bet': np.mean([h['bet_amount'] for h in hands])
            },
            'hands': self.current_session['hands']
        }

        # Guardar como JSON
        filename = self.save_dir / f"session_{summary['session_id']}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)

        # También guardar versión simplificada para CSV
        csv_filename = self.save_dir / f"session_{summary['session_id']}.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'state', 'action', 'reward', 'result', 'bet_amount'])
            for hand in hands:
                writer.writerow([
                    hand['timestamp'],
                    json.dumps(hand['state']),
                    hand['action'],
                    hand['reward'],
                    hand['result'],
                    hand['bet_amount']
                ])

        print(f"\n✅ Sesión guardada: {filename}")
        print(f"   Manos: {total_hands}")
        print(f"   Win rate: {summary['statistics']['win_rate']:.2%}")
        print(f"   Profit: ${total_profit:.2f}")

        # Reset para nueva sesión
        self.current_session = {'session_id': None, 'start_time': None, 'hands': [], 'casino_name': '', 'table_rules': {}}

        return filename


class ManualDataEntry:
    """Entrada manual de datos desde casino online/real."""

    def __init__(self):
        self.collector = CasinoDataCollector()

    def interactive_session(self):
        """Sesión interactiva para registrar manos."""
        print("\n" + "="*60)
        print("🎰 RECOPILADOR DE DATOS DE CASINO")
        print("="*60)

        casino = input("\nNombre del casino (o Enter para 'Online'): ") or "Online"
        num_decks = input("Número de mazos [6]: ") or "6"

        rules = {
            'num_decks': int(num_decks),
            'dealer_hits_soft_17': input("Dealer hits soft 17? [y/N]: ").lower() == 'y',
            'allow_surrender': input("Permite surrender? [y/N]: ").lower() == 'y',
            'blackjack_pays': input("Blackjack paga (3:2 o 6:5) [3:2]: ") or "3:2"
        }

        self.collector.start_session(casino, **rules)

        print("\n📝 Registrar manos (Enter sin valor para terminar):")
        print("Formato: tu_total,dealer_card,accion,resultado,apuesta")
        print("Ejemplo: 18,7,stand,win,10")

        while True:
            try:
                entry = input("\nMano: ").strip()
                if not entry:
                    break

                parts = entry.split(',')
                if len(parts) < 4:
                    print("❌ Formato incorrecto. Intenta de nuevo.")
                    continue

                # Simplificado: registrar datos básicos
                self.collector.record_hand(
                    state=[int(parts[0]), int(parts[1])],  # Simplificado
                    action=parts[2],
                    reward=+1 if parts[3] == 'win' else (-1 if parts[3] == 'loss' else 0),
                    result=parts[3],
                    bet_amount=float(parts[4]) if len(parts) > 4 else 10
                )
                print(f"✅ Registrado: {parts[0]} vs {parts[1]} → {parts[2]} → {parts[3]}")

            except Exception as e:
                print(f"❌ Error: {e}")

        # Terminar sesión
        self.collector.end_session()


# ============ USO ============

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        # Modo interactivo
        entry = ManualDataEntry()
        entry.interactive_session()

    else:
        # Modo programático (ejemplo)
        collector = CasinoDataCollector()

        # Iniciar sesión
        collector.start_session(
            casino_name="Casino Royale",
            num_decks=6,
            dealer_hits_soft_17=True,
            allow_surrender=True
        )

        # Ejemplo de registrar algunas manos
        # En uso real, esto vendría de tu juego
        print("\n📊 Ejemplo de uso programático:")
        print("En tu código de juego, llama a:")
        print("  collector.record_hand(state, action, reward, result, bet_amount)")
        print("\nPara terminar la sesión:")
        print("  collector.end_session()")
