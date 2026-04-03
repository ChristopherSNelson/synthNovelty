import json
from route_scorer import RouteScorer

def main():
    scorer = RouteScorer()

    # 1. Standard Route: Amide couplings (Common steps from our 'low novelty' set)
    standard_route = [
        "O[C:7](=[O:8])[C@H:9]([CH2:10][CH:11]1[CH2:12][CH2:13][CH2:14][CH2:15]1)[CH2:16][c:17]1[cH:18][cH:19][c:20]([F:21])[cH:22][cH:23]1.[NH2:1][C@@H:2]([CH3:3])[c:4]1[cH:5][cH:6][cH:24][cH:25][cH:26]1>>[CH3:3][C@H:2]([NH:1][C:7](=[O:8])[C@H:9]([CH2:10][CH:11]1[CH2:12][CH2:13][CH2:14][CH2:15]1)[CH2:16][c:17]1[cH:18][cH:19][c:20]([F:21])[cH:22][cH:23]1)[c:4]1[cH:5][cH:6][cH:24][cH:25][cH:26]1",
        "O[C:16]([CH:3]([O:2][CH3:1])[C:4](=[O:5])[NH:6][CH2:7][c:8]1[cH:9][c:10]([F:11])[cH:12][c:13]([F:14])[cH:15]1)=[O:17].[NH2:18][CH2:19][c:20]1[cH:21][cH:22][c:23]([Cl:24])[cH:25][cH:26]1>>[CH3:1][O:2][CH:3]([C:4](=[O:5])[NH:6][CH2:7][c:8]1[cH:9][c:10]([F:11])[cH:12][c:13]([F:14])[cH:15]1)[C:16](=[O:17])[NH:18][CH2:19][c:20]1[cH:21][cH:22][c:23]([Cl:24])[cH:25][cH:26]1"
    ]

    # 2. Novel Route: Using unusual transformations from our "High Novelty" set
    novel_route = [
        "N[NH:7][c:6]1[c:4]([Br:5])[cH:3][c:2]([CH3:1])[cH:11][cH:10]1.O=[C:8]1[CH2:9][CH2:12][CH2:13][CH2:14][CH2:15]1>>[CH3:1][c:2]1[cH:3][c:4]([Br:5])[c:6]2[n:7][n:8][c:9]3[CH2:12][CH2:13][CH2:14][CH2:15][c:8]3[n:10]2[cH:11]1",
        "O=C(OCC1c2ccccc2-c2ccccc21)[NH:37][CH2:36][CH2:35][NH:34][CH2:33][C@@H:32]1[CH2:31][O:30][C@@:27]([CH3:26])([C:23]([O:22][CH2:21][c:20]2[cH:19][cH:18][cH:17][cH:16][cH:15]2)=[O:24])[CH2:28][CH2:29]1>>[CH3:26][C@:27]1([C:23]([O:22][CH2:21][c:20]2[cH:19][cH:18][cH:17][cH:16][cH:15]2)=[O:24])[CH2:28][CH2:29][C@H:32]([CH2:33][NH:34][CH2:35][CH2:36][NH2:37])[CH2:31][O:30]1"
    ]

    print("\n" + "="*80)
    print("MULTI-STEP ROUTE NOVELTY DEMO")
    print("="*80)

    for name, route in [("Standard Route", standard_route), ("Novel Route", novel_route)]:
        results = scorer.score_route(route)
        
        print(f"\n>>> {name.upper()}")
        print(f"Number of steps: {results['num_steps']}")
        print(f"Mean Novelty:    {results['mean_novelty']:.4f}")
        print(f"Max Novelty:     {results['max_novelty']:.4f} (Bottleneck)")
        
        print("\nStep-by-step Novelty:")
        for i, (rxn, score) in enumerate(zip(route, results['step_scores'])):
            display_rxn = rxn if len(rxn) < 65 else rxn[:62] + "..."
            print(f"  Step {i+1}: {score:.4f} | {display_rxn}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("The 'Max Novelty' metric effectively identifies the most unusual")
    print("transformation in a sequence, allowing researchers to pinpoint")
    print("strategic innovation in multi-step retrosynthesis.")

if __name__ == "__main__":
    main()
