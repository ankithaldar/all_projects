#!/bin/bash

# usage in github codespaces to create the VSCode environment that I use

# pip install -r requirements.txt


code_extension=("arcticicestudio.nord-visual-studio-code" "bierner.markdown-mermaid" "donjayamanne.githistory" "giovdk21.vscode-sublime-merge" "GitHub.remotehub" "GitHub.vscode-pull-request-github" "mohsen1.prettify-json" "ms-python.python" "ms-python.vscode-pylance" "ms-toolsai.jupyter" "ms-toolsai.jupyter-keymap" "ms-toolsai.jupyter-renderers" "ms-vscode.cpptools" "ms-vscode.remote-repositories" "ms-vscode.sublime-keybindings" "redhat.vscode-yaml" "TabNine.tabnine-vscode" "wraith13.bracket-lens" "vscode-icons-team.vscode-icons" "EliverLara.andromeda")

for i in ${!code_extension[@]};
do
 code --install-extension ${code_extension[$i]} --force
done


# mkdir -p charts
