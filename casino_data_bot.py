"""
Bot para recopilar datos REALES de casinos online DEMO.

⚠️  IMPORTANTE:
- Solo usar en versiones DEMO/GRATIS (sin dinero real)
- No usar en casinos con dinero real (violación TOS)
- Respetar siempre los Términos de Servicio del casino

Uso:
1. Abre un casino online DEMO en tu navegador
2. Ejecuta este script
3. El script te pedirá ingresar los datos de cada mano
4. O usa modo automático con Selenium (para casinos compatibles)
"""

import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class CasinoDataBot:
    """Bot para recopilar datos de casinos online DEMO."""

    def __init__(self, save_dir='casino_data'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.current_session = {
            'casino_name': '',
            'casino_type': '',  # 'fiat' or 'crypto'
            'is_demo': True,
            'start_time': None,
            'hands': [],
            'metadata': {}
        }

    def start_session(self, casino_name, casino_type='fiat', is_demo=True):
        """Iniciar nueva sesión."""
        self.current_session.update({
            'casino_name': casino_name,
            'casino_type': casino_type,
            'is_demo': is_demo,
            'start_time': datetime.now().isoformat(),
            'hands': []
        })

        print(f"\n{'='*70}")
        print(f"🎰 SESIÓN INICIADA")
        print(f"{'='*70}")
        print(f"Casino: {casino_name}")
        print(f"Tipo: {casino_type.upper()}")
        print(f"Modo: {'DEMO (Gratis)' if is_demo else 'REAL (Dinero real)'} ⚠️")
        print(f"{'='*70}\n")

    def record_hand(self, player_total, dealer_card, action, result,
                    bet_amount=10, cards_player=None, cards_dealer=None,
                    split=False, doubled=False, blackjack=False):
        """Registrar una mano."""

        hand_data = {
            'timestamp': datetime.now().isoformat(),
            'player_total': int(player_total),
            'dealer_card': int(dealer_card),
            'action': action,
            'result': result,  # 'win', 'loss', 'push', 'blackjack'
            'bet_amount': float(bet_amount),
            'cards_player': cards_player,
            'cards_dealer': cards_dealer,
            'split': split,
            'doubled': doubled,
            'blackjack': blackjack
        }

        self.current_session['hands'].append(hand_data)

        # Mostrar resumen
        print(f"✅ Mano {len(self.current_session['hands'])}: "
              f"{player_total} vs {dealer_card} → {action} → {result}")

    def end_session(self):
        """Terminar sesión y guardar."""
        if not self.current_session['hands']:
            print("⚠️  No hay manos para guardar.")
            return None

        # Calcular estadísticas
        hands = self.current_session['hands']
        total_hands = len(hands)
        wins = sum(1 for h in hands if h['result'] in ['win', 'blackjack'])
        losses = sum(1 for h in hands if h['result'] == 'loss')
        pushes = sum(1 for h in hands if h['result'] == 'push')
        blackjacks = sum(1 for h in hands if h['result'] == 'blackjack')

        total_profit = sum([
            h['bet_amount'] if h['result'] in ['win', 'blackjack'] else
            (-h['bet_amount'] * 2 if h['blackjack'] else -h['bet_amount'])
            if h['result'] == 'loss' else 0
            for h in hands
        ])

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary = {
            'session_id': session_id,
            'casino_name': self.current_session['casino_name'],
            'casino_type': self.current_session['casino_type'],
            'is_demo': self.current_session['is_demo'],
            'start_time': self.current_session['start_time'],
            'end_time': datetime.now().isoformat(),
            'statistics': {
                'total_hands': total_hands,
                'wins': wins,
                'losses': losses,
                'pushes': pushes,
                'blackjacks': blackjacks,
                'win_rate': wins / total_hands if total_hands > 0 else 0,
                'total_profit': total_profit
            },
            'hands': hands
        }

        # Guardar
        filename = self.save_dir / f"{self.current_session['casino_type']}_{session_id}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print(f"📊 RESUMEN DE LA SESIÓN")
        print(f"{'='*70}")
        print(f"Total manos: {total_hands}")
        print(f"Victorias: {wins} ({wins/total_hands:.1%})")
        print(f"Derrotas: {losses} ({losses/total_hands:.1%})")
        print(f"Empates: {pushes} ({pushes/total_hands:.1%})")
        print(f"Blackjacks: {blackjacks}")
        print(f"Profit: ${total_profit:.2f}")
        print(f"Guardado en: {filename}")
        print(f"{'='*70}\n")

        return filename


class InteractiveBot:
    """Bot interactivo para registrar manos manualmente."""

    def __init__(self):
        self.bot = CasinoDataBot()

    def run(self):
        """Ejecutar bot interactivo."""
        print("\n" + "="*70)
        print("🤖 BOT DE RECOLECCIÓN DE DATOS - CASINOS ONLINE")
        print("="*70)

        # Configurar casino
        print("\n🎰 CONFIGURACIÓN DEL CASINO")
        casino = input("Nombre del casino (ej: 'BetOnline', 'Stake', 'BitStarz'): ")
        casino_type = input("Tipo [fiat/crypto]: ").lower() or 'fiat'

        if casino_type not in ['fiat', 'crypto']:
            print("❌ Tipo debe ser 'fiat' o 'crypto'")
            return

        # ⚠️ Advertencia
        if casino_type == 'crypto':
            print("\n⚠️  ¿Estás usando versión DEMO?")
            confirm = input("Confirmar que es DEMO [y/N]: ").lower()
            if confirm != 'y':
                print("❌ Solo puedes usar este bot en versiones DEMO")
                return

        # Iniciar sesión
        self.bot.start_session(casino, casino_type, is_demo=True)

        print("\n📝 Instrucciones:")
        print("   1. Abre el casino DEMO en tu navegador")
        print("   2. Juega normalmente")
        print("   3. Por cada mano, ingresa los datos aquí")
        print("   4. Enter sin valor para terminar\n")

        # Loop de registro
        hand_num = 1
        while True:
            print(f"\n--- Mano #{hand_num} ---")

            try:
                input_vals = input(f"Tu total, carta dealer, acción, resultado (Enter para terminar): ").strip()

                if not input_vals:
                    break

                parts = input_vals.split(',')
                if len(parts) < 4:
                    print("❌ Formato incorrecto. Ejemplo: 18,7,stand,win")
                    continue

                player_total = int(parts[0].strip())
                dealer_card = int(parts[1].strip())
                action = parts[2].strip().lower()
                result = parts[3].strip().lower()

                # Validar
                if action not in ['hit', 'stand', 'double', 'split', 'surrender']:
                    print(f"❌ Acción inválida: {action}")
                    continue

                if result not in ['win', 'loss', 'push', 'blackjack']:
                    print(f"❌ Resultado inválido: {result}")
                    continue

                # Opcional: apuesta
                bet = input("Apuesta (Enter para $10): ").strip()
                bet_amount = float(bet) if bet else 10

                # Registrar
                self.bot.record_hand(
                    player_total=player_total,
                    dealer_card=dealer_card,
                    action=action,
                    result=result,
                    bet_amount=bet_amount
                )

                hand_num += 1

            except ValueError as e:
                print(f"❌ Error: {e}")
                print("   Asegúrate de usar números para totales y apuestas")
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrumpido")
                break

        # Terminar sesión
        self.bot.end_session()


class SeleniumBot:
    """Bot automatizado con Selenium (para casinos DEMO compatibles)."""

    def __init__(self):
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium no instalado. Haz: pip install selenium")
        self.bot = CasinoDataBot()
        self.driver = None

    def start_driver(self):
        """Iniciar navegador Chrome."""
        from selenium.webdriver.chrome.options import Options

        options = Options()
        options.add_argument('--start-maximized')
        # options.add_argument('--headless')  # Comentar para ver el navegador

        self.driver = webdriver.Chrome(options=options)

    def detect_table_state(self):
        """
        Detectar estado de la mesa desde el navegador.

        ⚠️  ESTO ES UN EJEMPLO - Necesita adaptación a cada casino
        """
        try:
            # Detectar cartas del jugador
            player_cards = self.driver.find_elements(By.CLASS_NAME, "player-card")
            player_total = sum([int(card.text) for card in player_cards])

            # Detectar carta del dealer
            dealer_card = self.driver.find_element(By.CLASS_NAME, "dealer-upcard").text

            return {
                'player_total': player_total,
                'dealer_card': int(dealer_card),
                'cards': [card.text for card in player_cards]
            }
        except Exception as e:
            print(f"❌ Error detectando mesa: {e}")
            return None

    def click_action(self, action):
        """
        Hacer clic en una acción.

        ⚠️  ESTO ES UN EJEMPLO - Necesita adaptación a cada casino
        """
        action_buttons = {
            'hit': 'button-hit',
            'stand': 'button-stand',
            'double': 'button-double',
            'split': 'button-split'
        }

        try:
            button_id = action_buttons.get(action)
            if button_id:
                button = self.driver.find_element(By.ID, button_id)
                button.click()
                return True
        except Exception as e:
            print(f"❌ Error haciendo clic en {action}: {e}")
            return False

    def run_automated(self, casino_url, num_hands=100):
        """
        Ejecutar bot automatizado.

        ⚠️  REQUIERE ADAPTACIÓN a cada casino específico
        """
        print(f"\n🤖 Bot automatizado para {casino_url}")
        print(f"⚠️  Modo DEMO solamente")
        print(f"🎯 Objetivo: {num_hands} manos\n")

        self.start_driver()
        self.driver.get(casino_url)

        # Esperar a que el usuario inicie sesión manualmente
        input("🔐 Inicia sesión en el casino DEMO y presiona Enter...")

        # Nombre del casino
        casino_name = input("🎰 Nombre del casino: ")

        self.bot.start_session(casino_name, casino_type='fiat', is_demo=True)

        # Loop de juego
        for hand_num in range(num_hands):
            try:
                # Detectar estado
                state = self.detect_table_state()
                if not state:
                    print("❌ No se pudo detectar el estado. Abortando.")
                    break

                # Decidir acción (usando tu modelo entrenado o random)
                # action = decide_action(state)  # TODO: Implementar
                action = 'stand'  # Placeholder

                # Ejecutar acción
                self.click_action(action)

                # Esperar resultado
                time.sleep(2)

                # Detectar resultado y registrar
                # result = detect_result()  # TODO: Implementar
                result = 'push'  # Placeholder

                self.bot.record_hand(
                    player_total=state['player_total'],
                    dealer_card=state['dealer_card'],
                    action=action,
                    result=result,
                    bet_amount=10,
                    cards_player=state.get('cards')
                )

                print(f"✅ Mano {hand_num+1}/{num_hands} completada")

                # Esperar siguiente ronda
                time.sleep(3)

            except Exception as e:
                print(f"❌ Error en mano {hand_num+1}: {e}")
                break

        # Terminar
        self.bot.end_session()
        self.driver.quit()


# ============ LISTA DE CASINOS RECOMENDADOS (DEMO) ============

RECOMMENDED_CASINOS = {
    'fiat': [
        {
            'name': 'BetOnline',
            'demo_url': 'https://www.betonline.ag/casino',
            'notes': 'Tiene versión demo gratis'
        },
        {
            'name': 'Ignition Casino',
            'demo_url': 'https://www.ignitioncasino.eu/',
            'notes': 'Requiere registro pero tiene demo'
        },
        {
            'name': 'Bovada',
            'demo_url': 'https://www.bovada.lv/',
            'notes': 'Versión practice disponible'
        }
    ],
    'crypto': [
        {
            'name': 'Stake',
            'demo_url': 'https://stake.com/casino/games/blackjack',
            'notes': 'Tiene demo con ficticio coins'
        },
        {
            'name': 'BitStarz',
            'demo_url': 'https://bitstarz.com/',
            'notes': 'Modo play money disponible'
        },
        {
            'name': 'mBit Casino',
            'demo_url': 'https://www.mbitcasino.com/',
            'notes': 'Tiene versión demo'
        }
    ]
}


def print_casino_list():
    """Imprimir lista de casinos recomendados."""
    print("\n" + "="*70)
    print("🎰 CASINOS RECOMENDADOS (CON DEMO)")
    print("="*70)

    print("\n💵 FIAT CASINOS:")
    for casino in RECOMMENDED_CASINOS['fiat']:
        print(f"\n  🎲 {casino['name']}")
        print(f"     URL: {casino['demo_url']}")
        print(f"     Notas: {casino['notes']}")

    print("\n💎 CRYPTO CASINOS:")
    for casino in RECOMMENDED_CASINOS['crypto']:
        print(f"\n  🎲 {casino['name']}")
        print(f"     URL: {casino['demo_url']}")
        print(f"     Notas: {casino['notes']}")

    print("\n" + "="*70)


# ============ USO ============

if __name__ == "__main__":
    import sys

    print("""
╔════════════════════════════════════════════════════════════╗
║   🤖 BOT DE RECOLECCIÓN DE DATOS - CASINOS ONLINE         ║
║   ⚠️  USAR SOLO EN VERSIONES DEMO/GRATIS                  ║
╚════════════════════════════════════════════════════════════╝
    """)

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nModos disponibles:")
        print("  1. interactive  - Registrar manualmente mientras juegas")
        print("  2. automated   - Bot automatizado (requiere Selenium)")
        print("  3. list        - Ver lista de casinos recomendados")

        mode = input("\nSelecciona modo [interactive/automated/list]: ").lower()

    if mode == 'interactive' or mode == '1':
        bot = InteractiveBot()
        bot.run()

    elif mode == 'automated' or mode == '2':
        if not SELENIUM_AVAILABLE:
            print("\n❌ Selenium no instalado.")
            print("   Instala: pip install selenium")
            print("   Descarga ChromeDriver: https://chromedriver.chromium.org/")
        else:
            print("\n⚠️  EL MODO AUTOMATIZADO REQUIERE:")
            print("   1. Adaptación a cada casino específico")
            print("   2. Conocimiento de Selenium")
            print("   3. Uso RESPONSABLE y ÉTICO")
            print("\n   📖 Recomiendo empezar con modo 'interactive'")
            confirm = input("\n¿Continuar con modo automatizado? [y/N]: ").lower()
            if confirm == 'y':
                url = input("URL del casino DEMO: ")
                num_hands = int(input("Número de manos: "))
                bot = SeleniumBot()
                bot.run_automated(url, num_hands)

    elif mode == 'list' or mode == '3':
        print_casino_list()

    else:
        print(f"❌ Modo no válido: {mode}")
