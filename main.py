#!/usr/bin/env python3
"""
VibeB2B Main Control Script

Interactive CLI for managing all application services:
- Recall Flask server (meeting recording/transcription)
- Mastra AI assistant server
- Bot creation and management

Usage: python3 main.py
"""

import subprocess
import sys
import os
import signal
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global process tracking
processes = {
    'recall_server': None,
    'mastra_server': None
}

def print_header():
    """Print the application header"""
    print("\n" + "="*60)
    print("🎯 VibeB2B Control Center")
    print("="*60)
    print("Available commands:")
    print("  bot       - Create a meeting recording bot")
    print("  server    - Start/stop Recall Flask server")
    print("  mastra    - Start/stop Mastra AI server")
    print("  health    - Check server health")
    print("  status    - Show current process status")
    print("  kill      - Kill all running processes")
    print("  exit/quit - Exit the application")
    print("="*60)

def check_server_health(port, name):
    """Check if a server is running on the given port"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        if response.status_code == 200:
            return f"✅ {name} running on port {port}"
        else:
            return f"⚠️  {name} responding with status {response.status_code}"
    except:
        return f"❌ {name} not responding on port {port}"

def start_recall_server():
    """Start the Recall Flask server"""
    global processes

    if processes['recall_server'] and processes['recall_server'].poll() is None:
        print("ℹ️  Recall server is already running")
        return

    print("🚀 Starting Recall Flask server...")
    try:
        # Change to src/recall directory and start server
        process = subprocess.Popen(
            [sys.executable, 'src/recall/recallAPI.py'],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        processes['recall_server'] = process

        # Give it a moment to start
        time.sleep(2)

        if process.poll() is None:
            print("✅ Recall server started successfully")
        else:
            print("❌ Failed to start Recall server")
            processes['recall_server'] = None

    except Exception as e:
        print(f"❌ Error starting Recall server: {e}")
        processes['recall_server'] = None

def stop_recall_server():
    """Stop the Recall Flask server"""
    global processes

    if processes['recall_server'] and processes['recall_server'].poll() is None:
        print("🛑 Stopping Recall server...")
        processes['recall_server'].terminate()

        # Wait for it to terminate gracefully
        try:
            processes['recall_server'].wait(timeout=5)
            print("✅ Recall server stopped")
        except subprocess.TimeoutExpired:
            print("⚠️  Force killing Recall server...")
            processes['recall_server'].kill()
            processes['recall_server'].wait()
            print("✅ Recall server force killed")

        processes['recall_server'] = None
    else:
        print("ℹ️  Recall server is not running")

def start_mastra_server():
    """Start the Mastra AI server"""
    global processes

    if processes['mastra_server'] and processes['mastra_server'].poll() is None:
        print("ℹ️  Mastra server is already running")
        return

    print("🤖 Starting Mastra AI server...")
    try:
        process = subprocess.Popen(
            ['npm', 'run', 'dev'],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        processes['mastra_server'] = process

        # Give it a moment to start
        time.sleep(3)

        if process.poll() is None:
            print("✅ Mastra server started successfully")
        else:
            print("❌ Failed to start Mastra server")
            processes['mastra_server'] = None

    except Exception as e:
        print(f"❌ Error starting Mastra server: {e}")
        processes['mastra_server'] = None

def stop_mastra_server():
    """Stop the Mastra AI server"""
    global processes

    if processes['mastra_server'] and processes['mastra_server'].poll() is None:
        print("🛑 Stopping Mastra server...")
        processes['mastra_server'].terminate()

        # Wait for it to terminate gracefully
        try:
            processes['mastra_server'].wait(timeout=5)
            print("✅ Mastra server stopped")
        except subprocess.TimeoutExpired:
            print("⚠️  Force killing Mastra server...")
            processes['mastra_server'].kill()
            processes['mastra_server'].wait()
            print("✅ Mastra server force killed")

        processes['mastra_server'] = None
    else:
        print("ℹ️  Mastra server is not running")

def create_bot():
    """Create a meeting recording bot"""
    meeting_url = input("📹 Enter meeting URL: ").strip()

    if not meeting_url:
        print("❌ No meeting URL provided")
        return

    print(f"🎯 Creating bot for meeting: {meeting_url}")

    # Call the Recall API directly
    try:
        response = requests.post(
            "http://localhost:3000/start_VB2B",
            json={"meeting_url": meeting_url},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Bot created successfully!")
            print(f"   Bot ID: {result.get('id', 'Unknown')}")
            print(f"   Meeting: {meeting_url}")
            if 'meeting_url' in result and 'platform' in result['meeting_url']:
                print(f"   Platform: {result['meeting_url']['platform']}")
        else:
            print(f"❌ Failed to create bot: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Recall server. Make sure it's running with 'server' command.")
    except Exception as e:
        print(f"❌ Error creating bot: {e}")

def show_status():
    """Show status of all services"""
    print("\n📊 Service Status:")
    print("-" * 30)

    # Check Recall server
    recall_status = check_server_health(3000, "Recall Server")
    print(recall_status)

    # Check Mastra server (usually runs on a different port, check common ones)
    mastra_ports = [3001, 5173, 4000, 8000]
    mastra_found = False
    for port in mastra_ports:
        try:
            response = requests.get(f"http://localhost:{port}", timeout=1)
            print(f"✅ Mastra Server running on port {port}")
            mastra_found = True
            break
        except:
            continue

    if not mastra_found:
        print("❌ Mastra Server not found on common ports")

    # Show process status
    print("\n🔧 Process Status:")
    print(f"   Recall Server: {'✅ Running' if processes['recall_server'] and processes['recall_server'].poll() is None else '❌ Not running'}")
    print(f"   Mastra Server: {'✅ Running' if processes['mastra_server'] and processes['mastra_server'].poll() is None else '❌ Not running'}")

def kill_all_processes():
    """Kill all running processes"""
    print("🧹 Killing all processes...")

    stop_recall_server()
    stop_mastra_server()

    print("✅ All processes stopped")

def cleanup_processes():
    """Cleanup function called on exit"""
    kill_all_processes()

def main():
    """Main interactive loop"""
    # Register cleanup on exit
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup_processes())
    signal.signal(signal.SIGTERM, lambda sig, frame: cleanup_processes())

    print_header()

    while True:
        try:
            command = input("\n🎯 Enter command: ").strip().lower()

            if command in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                cleanup_processes()
                break

            elif command == 'bot':
                create_bot()

            elif command == 'server':
                action = input("Start or stop Recall server? (start/stop): ").strip().lower()
                if action == 'start':
                    start_recall_server()
                elif action == 'stop':
                    stop_recall_server()
                else:
                    print("❌ Invalid action. Use 'start' or 'stop'")

            elif command == 'mastra':
                action = input("Start or stop Mastra server? (start/stop): ").strip().lower()
                if action == 'start':
                    start_mastra_server()
                elif action == 'stop':
                    stop_mastra_server()
                else:
                    print("❌ Invalid action. Use 'start' or 'stop'")

            elif command == 'health':
                show_status()

            elif command == 'status':
                show_status()

            elif command == 'kill':
                kill_all_processes()

            elif command == 'help':
                print_header()

            else:
                print(f"❌ Unknown command: {command}")
                print("Type 'help' for available commands")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            cleanup_processes()
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
