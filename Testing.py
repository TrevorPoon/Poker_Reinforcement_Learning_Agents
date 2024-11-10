def get_hand_type(self, hole_card):

    card_on_hand = hole_card
    # Sort cards by rank (for consistency)
    ranks = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    card1, card2 = sorted(card_on_hand, key=lambda card: ranks[card[1]])

    # Extract ranks and suits
    rank1, suit1 = card1[1], card1[0]
    rank2, suit2 = card2[1], card2[0]

    # Determine hand type
    if rank1 == rank2:
        return rank1 + rank2  # e.g., "AA" for pocket aces
    elif suit1 == suit2:
        return rank1 + rank2 + 's'  # e.g., "AKs" for suited
    else:
        return rank1 + rank2 + 'o'  # e.g., "AKo" for offsuit
    
# self.hand_type = self.get_hand_type(hole_card)

self.card_reward_stat['hands'][self.hand_type] += reward