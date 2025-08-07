#!/usr/bin/env python3
"""
Documentation build and serve script for Alinemol.

This script provides convenient commands to build, serve, and deploy
the MkDocs documentation.

Usage:
    python build_docs.py [command] [options]

Commands:
    build    - Build the documentation
    serve    - Serve documentation locally
    clean    - Clean build artifacts
    deploy   - Deploy to GitHub Pages

Examples:
    python build_docs.py serve          # Serve locally at http://localhost:8000
    python build_docs.py build          # Build static site to site/
    python build_docs.py deploy         # Deploy to GitHub Pages
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=check, capture_output=False)
        print(f"✅ {description} completed successfully")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print(f"\n⚠️ {description} interrupted by user")
        return 130


def check_mkdocs_available():
    """Check if MkDocs is installed."""
    try:
        result = subprocess.run(["mkdocs", "--version"], check=True, capture_output=True, text=True)
        print(f"✅ MkDocs available: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ MkDocs not found. Install with: pip install -e '.[docs]'")
        return False


def build_docs(args):
    """Build the documentation."""
    if not check_mkdocs_available():
        return 1

    # Check if we're in the right directory
    if not Path("mkdocs.yml").exists():
        print("❌ Error: mkdocs.yml not found. Please run from project root.")
        return 1

    cmd = ["mkdocs", "build"]

    if args.clean:
        cmd.append("--clean")

    if args.strict:
        cmd.append("--strict")

    if args.verbose:
        cmd.append("--verbose")

    return run_command(cmd, "Building documentation")


def serve_docs(args):
    """Serve documentation locally."""
    if not check_mkdocs_available():
        return 1

    if not Path("mkdocs.yml").exists():
        print("❌ Error: mkdocs.yml not found. Please run from project root.")
        return 1

    cmd = ["mkdocs", "serve"]

    if args.dev_addr:
        cmd.extend(["--dev-addr", args.dev_addr])

    if args.livereload:
        cmd.append("--livereload")
    elif args.no_livereload:
        cmd.append("--no-livereload")

    print("🌐 Starting development server...")
    print(f"📖 Documentation will be available at: http://{args.dev_addr or '127.0.0.1:8000'}")
    print(f"🔄 Live reload: {'enabled' if args.livereload else 'disabled'}")
    print("⏹️  Press Ctrl+C to stop the server")

    return run_command(cmd, "Serving documentation", check=False)


def clean_docs(args):
    """Clean documentation build artifacts."""
    artifacts = ["site/", ".mkdocs_cache/"]

    print("🧹 Cleaning documentation artifacts...")

    for artifact in artifacts:
        artifact_path = Path(artifact)
        if artifact_path.exists():
            if artifact_path.is_dir():
                shutil.rmtree(artifact_path)
                print(f"  🗑️  Removed directory: {artifact}")
            else:
                artifact_path.unlink()
                print(f"  🗑️  Removed file: {artifact}")
        else:
            print(f"  ✅ Already clean: {artifact}")

    print("✅ Documentation cleanup completed")
    return 0


def deploy_docs(args):
    """Deploy documentation to GitHub Pages."""
    if not check_mkdocs_available():
        return 1

    if not Path("mkdocs.yml").exists():
        print("❌ Error: mkdocs.yml not found. Please run from project root.")
        return 1

    # Check if we're in a git repository
    try:
        subprocess.run(["git", "status"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ Error: Not in a git repository or git not available.")
        return 1

    # Check for uncommitted changes
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)

    if result.stdout.strip() and not args.force:
        print("⚠️  Warning: You have uncommitted changes.")
        print("   Commit your changes first or use --force to deploy anyway.")
        print("   Uncommitted files:")
        for line in result.stdout.strip().split("\n"):
            print(f"     {line}")
        return 1

    cmd = ["mkdocs", "gh-deploy"]

    if args.force:
        cmd.append("--force")

    if args.message:
        cmd.extend(["--message", args.message])

    return run_command(cmd, "Deploying to GitHub Pages")


def validate_docs(args):
    """Validate documentation structure and links."""
    print("🔍 Validating documentation structure...")

    # Check required files
    required_files = [
        "mkdocs.yml",
        "docs/index.md",
        "docs/user-guide/getting-started.md",
        "docs/api/distance.md",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required documentation files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return 1

    # Try building with strict mode
    print("🔧 Testing build with strict mode...")
    result = subprocess.run(["mkdocs", "build", "--strict"], capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Documentation build failed in strict mode:")
        print(result.stderr)
        return 1

    print("✅ Documentation validation passed")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Build and serve Alinemol documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build documentation")
    build_parser.add_argument("--clean", action="store_true", help="Clean before building")
    build_parser.add_argument("--strict", action="store_true", help="Strict mode (fail on warnings)")
    build_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve documentation locally")
    serve_parser.add_argument("--dev-addr", default="127.0.0.1:8000", help="Development server address")
    serve_parser.add_argument("--livereload", action="store_true", help="Enable live reload")
    serve_parser.add_argument("--no-livereload", action="store_true", help="Disable live reload")

    # Clean command
    subparsers.add_parser("clean", help="Clean build artifacts")

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to GitHub Pages")
    deploy_parser.add_argument("--force", action="store_true", help="Force deployment even with uncommitted changes")
    deploy_parser.add_argument("--message", help="Deployment commit message")

    # Validate command
    subparsers.add_parser("validate", help="Validate documentation")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    if args.command == "build":
        return build_docs(args)
    elif args.command == "serve":
        return serve_docs(args)
    elif args.command == "clean":
        return clean_docs(args)
    elif args.command == "deploy":
        return deploy_docs(args)
    elif args.command == "validate":
        return validate_docs(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
