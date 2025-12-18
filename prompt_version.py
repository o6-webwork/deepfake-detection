#!/usr/bin/env python3
"""
Prompt Version Management Tool (File-per-Version System)

This script manages prompt versions stored as separate files in prompts/ directory:
- Each version is stored as prompts/v{version}.yaml (immutable)
- prompts/versions.json tracks version registry and metadata
- prompts/current.yaml is the active version (symlink or copy)

Usage:
    python prompt_version.py info                    # Show current version info
    python prompt_version.py list                    # List all versions
    python prompt_version.py changelog               # Show full changelog
    python prompt_version.py bump [major|minor|patch] # Create new version
    python prompt_version.py activate <version>      # Switch to different version
    python prompt_version.py diff <v1> <v2>          # Compare two versions
"""

import yaml
import json
import sys
import shutil
from datetime import datetime
from pathlib import Path


class PromptVersionManager:
    """Manage prompt versions in prompts/ directory"""

    def __init__(self, prompts_dir="prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.versions_file = self.prompts_dir / "versions.json"
        self.current_file = self.prompts_dir / "current.yaml"

        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

        if not self.versions_file.exists():
            raise FileNotFoundError(f"Version registry not found: {self.versions_file}")

        # Load version registry
        with open(self.versions_file, 'r', encoding='utf-8') as f:
            self.registry = json.load(f)

    def _save_registry(self):
        """Save version registry to disk."""
        with open(self.versions_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def get_current_version(self):
        """Get current active version string."""
        return self.registry.get('current_version', 'unknown')

    def get_version_info(self, version):
        """Get metadata for specific version."""
        return self.registry['versions'].get(version)

    def list_versions(self):
        """List all available versions."""
        print("=" * 60)
        print("Available Prompt Versions")
        print("=" * 60)

        current = self.get_current_version()
        versions = sorted(self.registry['versions'].keys(), reverse=True)

        for version in versions:
            info = self.get_version_info(version)
            active_marker = " (ACTIVE)" if version == current else ""
            print(f"\n📌 Version {version}{active_marker}")
            print(f"   File:   {info['file']}")
            print(f"   Date:   {info['date']}")
            print(f"   Author: {info['author']}")
            print(f"   Changes:")
            for change in info['changes']:
                print(f"     - {change}")

        print("=" * 60)

    def show_info(self):
        """Display current version information."""
        current = self.get_current_version()
        info = self.get_version_info(current)

        print("=" * 60)
        print("NexInspect Prompt Version Information")
        print("=" * 60)
        print(f"Current Version: {current}")
        print(f"Description:     {self.registry.get('description', 'N/A')}")
        print(f"Active File:     {info['file']}")
        print(f"Last Updated:    {info['date']}")
        print(f"Author:          {info['author']}")
        print()
        print("Latest Changes:")
        for change in info['changes']:
            print(f"  - {change}")
        print("=" * 60)

    def show_changelog(self):
        """Display full changelog across all versions."""
        print("=" * 60)
        print("Prompt Version Changelog")
        print("=" * 60)

        # Sort versions in descending order
        versions = sorted(self.registry['versions'].keys(),
                         key=lambda v: [int(x) for x in v.split('.')],
                         reverse=True)

        for version in versions:
            info = self.get_version_info(version)
            print(f"\nVersion {version} - {info['date']}")
            print(f"Author: {info['author']}")
            print("Changes:")
            for change in info['changes']:
                print(f"  - {change}")
            print("-" * 60)

    def bump_version(self, bump_type='patch', changes=None, author='User'):
        """
        Create a new version by bumping the version number.

        Args:
            bump_type: 'major', 'minor', or 'patch'
            changes: List of change descriptions
            author: Author name

        Returns:
            New version string
        """
        current_version = self.get_current_version()

        # Parse current version
        try:
            major, minor, patch = map(int, current_version.split('.'))
        except ValueError:
            print(f"Error: Invalid version format: {current_version}")
            return None

        # Bump version
        if bump_type == 'major':
            major += 1
            minor = 0
            patch = 0
        elif bump_type == 'minor':
            minor += 1
            patch = 0
        elif bump_type == 'patch':
            patch += 1
        else:
            print(f"Error: Invalid bump type: {bump_type}")
            return None

        new_version = f"{major}.{minor}.{patch}"
        today = datetime.now().strftime('%Y-%m-%d')

        # Check if version already exists
        if new_version in self.registry['versions']:
            print(f"Error: Version {new_version} already exists")
            return None

        # Copy current version file to new version file
        current_file = self.prompts_dir / self.get_version_info(current_version)['file']
        new_file = self.prompts_dir / f"v{new_version}.yaml"

        shutil.copy2(current_file, new_file)

        # Register new version
        self.registry['versions'][new_version] = {
            'file': f"v{new_version}.yaml",
            'date': today,
            'author': author,
            'changes': changes or ['Version bump'],
            'active': False
        }

        print(f"✓ Created new version: {current_version} → {new_version}")
        print(f"  File: {new_file}")
        print(f"  Date: {today}")
        print(f"  Author: {author}")
        print("  Changes:")
        for change in (changes or ['Version bump']):
            print(f"    - {change}")
        print()
        print(f"⚠️  New version created but not activated.")
        print(f"   Edit {new_file} with your changes, then run:")
        print(f"   python prompt_version.py activate {new_version}")

        self._save_registry()
        return new_version

    def activate(self, version):
        """
        Switch to a different version.

        Args:
            version: Version string to activate (e.g., "1.0.0")
        """
        if version not in self.registry['versions']:
            print(f"Error: Version {version} not found")
            print(f"Available versions: {', '.join(sorted(self.registry['versions'].keys()))}")
            return False

        # Deactivate current version
        current = self.get_current_version()
        if current in self.registry['versions']:
            self.registry['versions'][current]['active'] = False

        # Activate new version
        self.registry['versions'][version]['active'] = True
        self.registry['current_version'] = version

        # Copy version file to current.yaml
        version_file = self.prompts_dir / self.get_version_info(version)['file']
        shutil.copy2(version_file, self.current_file)

        print(f"✓ Activated version {version}")
        print(f"  {version_file} → current.yaml")

        self._save_registry()
        return True

    def diff(self, version1, version2):
        """
        Compare two versions (prints instruction for now).

        Args:
            version1: First version string
            version2: Second version string
        """
        if version1 not in self.registry['versions']:
            print(f"Error: Version {version1} not found")
            return
        if version2 not in self.registry['versions']:
            print(f"Error: Version {version2} not found")
            return

        file1 = self.prompts_dir / self.get_version_info(version1)['file']
        file2 = self.prompts_dir / self.get_version_info(version2)['file']

        print(f"To compare versions {version1} and {version2}, run:")
        print(f"  diff {file1} {file2}")
        print()
        print("Or use your preferred diff tool:")
        print(f"  code --diff {file1} {file2}")
        print(f"  vimdiff {file1} {file2}")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    try:
        manager = PromptVersionManager()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if command == 'info':
        manager.show_info()

    elif command == 'list':
        manager.list_versions()

    elif command == 'changelog':
        manager.show_changelog()

    elif command == 'bump':
        bump_type = sys.argv[2] if len(sys.argv) > 2 else 'patch'

        # Collect changes from command line or prompt user
        changes = []
        if len(sys.argv) > 3:
            # Accept changes from remaining args
            changes = sys.argv[3:]
        else:
            print("Enter change descriptions (one per line, empty line to finish):")
            while True:
                change = input("  - ")
                if not change:
                    break
                changes.append(change)

        author = input("Author name (default: User): ").strip() or "User"
        manager.bump_version(bump_type, changes or None, author)

    elif command == 'activate':
        if len(sys.argv) < 3:
            print("Error: Version argument required")
            print("Usage: python prompt_version.py activate <version>")
            sys.exit(1)
        version = sys.argv[2]
        manager.activate(version)

    elif command == 'diff':
        if len(sys.argv) < 4:
            print("Error: Two version arguments required")
            print("Usage: python prompt_version.py diff <version1> <version2>")
            sys.exit(1)
        version1 = sys.argv[2]
        version2 = sys.argv[3]
        manager.diff(version1, version2)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()
