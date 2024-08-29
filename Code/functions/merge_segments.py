#Made by ChatGPT
def merge_segments(segments):
    if not segments:
        return []

    result = [segments[0]]

    for segment in segments[1:]:
        last_segment = result[-1]

        # Merge segments if they overlap or are adjacent
        if last_segment[1] + 1 >= segment[0]:
            last_segment = (last_segment[0], max(last_segment[1], segment[1]))
            result[-1] = last_segment
        else:
            # Add the segment to the list if they are disjoint
            result.append(segment)

    return result


if __name__ == '__main__':
    # Example of usage:
    segments_input1 = [(1, 3), (2, 5), (6, 8), (9, 12)]
    segments_input2 = [(1, 3), (2, 5), (7, 8), (7, 12)]
    result = merge_segments(segments_input1)
    print(result)