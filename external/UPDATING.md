# Updating

To update to a new snapshot, do something like:

    mv llama.cpp old
    git clone --depth 1 git@github.com:ggerganov/llama.cpp.git
    cd llama.cpp
    git log -n 1 > SNAPSHOT
    rm -rf .git
    rm -rf .github
    rm -rf .devops

Now fix anything that's wrong and add any custom patches.

Now you can remove the `old` folder.
