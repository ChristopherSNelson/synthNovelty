import torch
import numpy as np
import json
from score_reaction import load_model, score_reactions

class RouteScorer:
    def __init__(self):
        print("Loading RouteScorer (this may take a moment)...")
        (self.model, self.mean, self.std, 
         self.rxnfp_generator, self.mean_freq, self.device) = load_model()

    def score_route(self, reactions):
        """
        Score a route consisting of a list of reaction SMILES.
        Returns a dictionary with various pooling metrics.
        """
        if not reactions:
            return None
        
        step_scores = score_reactions(
            reactions, self.model, self.mean, self.std, 
            self.rxnfp_generator, self.mean_freq, self.device
        )
        
        return {
            "step_scores": step_scores.tolist(),
            "max_novelty": float(np.max(step_scores)),
            "mean_novelty": float(np.mean(step_scores)),
            "min_novelty": float(np.min(step_scores)),
            "num_steps": len(reactions)
        }

    def score_tree(self, tree_json):
        """
        Recursively extract reactions from a tree-like JSON structure
        (common in ASKCOS or AiZynthFinder outputs) and score them.
        """
        reactions = []
        
        def extract_reactions(node):
            # Check for common formats:
            # format 1 (Askcos/Aizynthfinder-like): is_chemical / is_reaction
            if isinstance(node, dict):
                if node.get("is_reaction") or node.get("type") == "reaction":
                    if "smiles" in node:
                        reactions.append(node["smiles"])
                
                # Recurse through children
                children = node.get("children", [])
                for child in children:
                    extract_reactions(child)
            elif isinstance(node, list):
                for item in node:
                    extract_reactions(item)

        extract_reactions(tree_json)
        return self.score_route(reactions)

if __name__ == "__main__":
    # Quick internal test with a mock route
    scorer = RouteScorer()
    mock_route = [
        "CC(=O)Cl.NCC>>CC(=O)NCC",
        "CC(=O)NCC.BrBr>>CC(=O)N(Br)CC"
    ]
    results = scorer.score_route(mock_route)
    print("\nMock Route Results:")
    print(json.dumps(results, indent=2))
