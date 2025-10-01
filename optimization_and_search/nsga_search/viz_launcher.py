#!/usr/bin/env python3
"""
Evolution Visualization Launcher

Easy access to all visualization tools from the main directory.
"""

import os
import sys
import subprocess
import argparse

def find_log_files():
    """Find available log files"""
    log_files = []
    if os.path.exists("logs"):
        log_files = [f for f in os.listdir("logs") if f.endswith('.log')]
    return log_files

def create_static_plots(log_file=None, output_dir="plots"):
    """Create static matplotlib plots"""
    if log_file:
        # Pass log file to the visualization creation
        cmd = [sys.executable, "-c", f"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from visualization import create_evolution_plots
create_evolution_plots('{log_file}', '{output_dir}')
print('📊 Static plots created successfully!')
"""]
    else:
        # Use the default create_visualizations.py
        cmd = [sys.executable, "visualization/create_visualizations.py"]
    subprocess.run(cmd)

def serve_dashboard(log_file, port=6006):
    """Start dashboard server"""
    cmd = [sys.executable, "visualization/servers/quick_server.py", log_file, str(port)]
    subprocess.run(cmd)

def launch_interactive():
    """Launch interactive menu"""
    cmd = [sys.executable, "visualization/servers/launch_dashboard.py"]
    subprocess.run(cmd)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Evolution Visualization Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --plots                          # Create static plots
  %(prog)s --serve logs/run_latest.log      # Start dashboard server
  %(prog)s --interactive                    # Interactive menu
  %(prog)s --serve logs/run_latest.log --port 8080  # Custom port
        """
    )
    
    parser.add_argument("--plots", nargs='?', const=True, metavar="LOG_FILE", 
                        help="Create static plots (optionally specify log file)")
    parser.add_argument("--serve", metavar="LOG_FILE", help="Start dashboard server")
    parser.add_argument("--interactive", action="store_true", help="Interactive launcher")
    parser.add_argument("--port", type=int, default=6006, help="Server port (default: 6006)")
    
    args = parser.parse_args()
    
    print("🧬 Evolution Visualization Tools")
    print("=" * 40)
    
    if args.plots is not False:
        if isinstance(args.plots, str):
            # Specific log file provided
            print(f"📊 Creating static plots from: {args.plots}")
            create_static_plots(args.plots)
        else:
            # Use default behavior
            print("📊 Creating static plots...")
            create_static_plots()
        
    elif args.serve:
        print(f"🚀 Starting dashboard server for: {args.serve}")
        serve_dashboard(args.serve, args.port)
        
    elif args.interactive:
        print("🎮 Launching interactive menu...")
        launch_interactive()
        
    else:
        # Show available options
        log_files = find_log_files()
        
        print(f"📁 Found {len(log_files)} log file(s):")
        for log_file in log_files:
            print(f"   - {log_file}")
        
        print(f"\n🛠️  Available commands:")
        print(f"   python viz_launcher.py --plots                    # Use default log")
        print(f"   python viz_launcher.py --plots LOG_FILE           # Specific log file")
        print(f"   python viz_launcher.py --interactive")
        if log_files:
            latest_log = f"logs/{log_files[0]}"
            print(f"   python viz_launcher.py --serve {latest_log}")
        
        print(f"\n💡 For help: python viz_launcher.py --help")

if __name__ == "__main__":
    main()
