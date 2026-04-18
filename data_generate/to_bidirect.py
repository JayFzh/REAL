import os
import dgl
import torch

def convert_to_bidirectional_with_self_loop(input_dir, output_dir=None):
    """
    Convert all DGL graphs in a directory to undirected graphs and add self-loops.

    Args:
        input_dir: path to the directory containing input .dgl files
        output_dir: path to the output directory; if None, overwrites the input files
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.dgl'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"Processing file: {filename}")

            graphs, labels = dgl.load_graphs(input_path)

            converted_graphs = []
            for i, g in enumerate(graphs):
                print(f"  Original graph {i}: {g.num_nodes()} nodes, {g.num_edges()} edges")

                # Convert to bidirected graph
                g_bidirectional = dgl.to_bidirected(g, copy_ndata=True)

                # Add self-loops
                g_with_self_loop = dgl.add_self_loop(g_bidirectional)

                print(f"  Converted graph {i}: {g_with_self_loop.num_nodes()} nodes, {g_with_self_loop.num_edges()} edges")

                converted_graphs.append(g_with_self_loop)

            dgl.save_graphs(output_path, converted_graphs, labels)
            print(f"  Saved to: {output_path}\n")


def remove_feat_and_save(input_dir, output_dir=None):
    """
    Remove node and edge features from all DGL graphs in a directory and save.

    Args:
        input_dir: path to the directory containing input .dgl files
        output_dir: path to the output directory; if None, overwrites the input files
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.dgl'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"Processing file: {filename}")

            graphs, labels = dgl.load_graphs(input_path)

            converted_graphs = []
            for i, g in enumerate(graphs):
                print(f"  Original graph {i}: {g.num_nodes()} nodes, {g.num_edges()} edges")

                del g.ndata['feat']

                print(f"  Features removed from graph {i}: {g.num_nodes()} nodes, {g.num_edges()} edges")

                converted_graphs.append(g)

            dgl.save_graphs(output_path, converted_graphs, labels)
            print(f"  Saved to: {output_path}\n")


if __name__ == "__main__":
    input_directory = "YOUR_STK_GRAPHS_DIR"   # e.g. /path/to/graphs/stk/graphs/
    output_directory = "YOUR_STK_GRAPHS_DIR"  # set to a different path to avoid overwriting

    convert_to_bidirectional_with_self_loop(input_directory, output_directory)
    # remove_feat_and_save(input_directory, output_directory)
