import jetson.inference
import jetson.utils
import numpy
import matplotlib

import argparse
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.")

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


def run_model():
    # load the object detection network
    net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

    # create video sources & outputs
    input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
    output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

    # process frames until the user exits
    while True:
        # capture the next image
        img = input.Capture()

        # detect objects in the image (with overlay)
        detections = net.Detect(img, overlay=opt.overlay)

        # print the detections
        print("detected {:d} objects in image".format(len(detections)))

        for detection in detections:
            print(detection)

        # render the image
        output.Render(img)

        # update the title bar
        output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

        # print out performance info
        net.PrintProfilerTimes()

        # exit on input/output EOS
        if not input.IsStreaming() or not output.IsStreaming():
            break

run_model()

def dijkstra(current, nodes, distances):
    # These are all the nodes which have not been visited yet
    unvisited = {node: None for node in nodes}
    # It will store the shortest distance from one node to another
    visited = {}
    # It will store the predecessors of the nodes
    currentDistance = 0
    unvisited[current] = currentDistance
    # Running the loop while all the nodes have been visited
    while True:
        # iterating through all the unvisited node
        for neighbour, distance in distances[current].items():
            # Iterating through the connected nodes of current_node (for 
            # example, a is connected with b and c having values 10 and 3
            # respectively) and the weight of the edges
            if neighbour not in unvisited: continue
            newDistance = currentDistance + distance
            if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:
                unvisited[neighbour] = newDistance
        # Till now the shortest distance between the source node and target node 
        # has been found. Set the current node as the target node
        visited[current] = currentDistance
        del unvisited[current]
        if not unvisited: break
        candidates = [node for node in unvisited.items() if node[1]]
        print(sorted(candidates, key = lambda x: x[1]))
        current, currentDistance = sorted(candidates, key = lambda x: x[1])[0]
    return visited
  
nodes = ('A', 'B', 'C', 'D', 'E')
distances = {
    'A': {'B': 5, 'C': 2},
    'B': {'C': 2, 'D': 3},
    'C': {'B': 3, 'D': 7},
    'D': {'E': 7},
    'E': {'D': 9}}
current = 'A'
  
print(dijkstra(current, nodes, distances))