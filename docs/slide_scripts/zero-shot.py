from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='FourierGPT Zero-Shot Detection Process')
dot.attr(rankdir='LR', fontsize='12', fontname='Arial')  # Graph-wide font
dot.node_attr.update(fontname='Arial')  # Apply Arial to all nodes
dot.edge_attr.update(fontname='Arial')  # Apply Arial to all edges

# Define the steps
dot.node('A', 'Compute NLL scores')
dot.node('B', 'Normalize NLL scores\n(using z-score)')
dot.node('C', 'Apply Fourier Transform\n(to get spectrum)')
dot.node('D', 'Classify using heuristic\n(power sum of first k frequencies)\n(k tuned on validation)')

# Connect the steps
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')

# Save to file
output_path = './slide_imgs/fouriergpt_process'
dot.render(output_path, format='png', cleanup=True)

print(f"Process graph saved to {output_path}.png")
