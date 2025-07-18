#!/bin/bash

# Update package lists
sudo apt update

# Create a cleaned package list
grep -vE '^(libasound2|libicu66|python3.8|llvm-10)' systempackages.txt > cleaned_packages.txt

# Install with automatic replacements
xargs -a cleaned_packages.txt sudo apt install -y --allow-downgrades --fix-broken

# Verify installations
echo "Verifying installations..."
while read pkg; do
    if dpkg -s "$pkg" &> /dev/null || dpkg -s "${pkg/t64}" &> /dev/null; then
        echo "✅ $pkg installed"
    else
        echo "❌ $pkg FAILED - Try: sudo apt install ${pkg/t64}t64"
    fi
done < systempackages.txt