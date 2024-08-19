## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14

import collections
import datetime
import gc
import time
from copy import copy
from typing import List, Union, Optional

import torch
# import torch
import numpy as np

from util.logFile import log






def padded_tensor(
        items: List[Union[List[int], torch.LongTensor]],
        pad_idx: int = 0,
        pad_tail: bool = True,
        max_len: Optional[int] = None,
) -> torch.LongTensor:
    """
     from https://github.com/Zyh716/WSDM2022-C2CRS
    Create a padded matrix from an uneven list of lists.

    Returns padded matrix.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param int pad_idx: the value to use for padding
    :param bool pad_tail:
    :param int max_len: if None, the max length is the maximum item length

    :returns: padded tensor.
    :rtype: Tensor[int64]

    """
    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]  # type: ignore
    # max in time dimension
    t = max(lens) if max_len is None else max_len
    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)

    if isinstance(items[0], torch.Tensor):
        # keep type of input tensors, they may already be cuda ones
        output = items[0].new(n, t)  # type: ignore
    else:
        output = torch.LongTensor(n, t)  # type: ignore
    output.fill_(pad_idx)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            # skip empty items
            continue
        if not isinstance(item, torch.Tensor):
            # put non-tensors into a tensor
            item = torch.tensor(item, dtype=torch.long)  # type: ignore
        if pad_tail:
            # place at beginning
            output[i, :length] = item
        else:
            # place at end
            output[i, t - length:] = item

    return output



def add_start_end_token_idx(vec: list, start_token_idx: int = None, end_token_idx: int = None):
    """
    from https://github.com/Zyh716/WSDM2022-C2CRS
    Can choose to add start token in the beginning and end token in the end.

    Args:
        vec: source list composed of indexes.
        start_token_idx: index of start token.
        end_token_idx: index of end token.

    Returns:
        list: list added start or end token index.

    """
    res = copy(vec)
    if start_token_idx:
        res.insert(0, start_token_idx)
    if end_token_idx:
        res.append(end_token_idx)
    return res


def truncate(vec, max_length, truncate_tail=True):
    """
     from https://github.com/Zyh716/WSDM2022-C2CRS
    truncate vec to make its length no more than max length.

    Args:
        vec (list): source list.
        max_length (int)
        truncate_tail (bool, optional): Defaults to True.

    Returns:
        list: truncated vec.

    """
    if max_length is None:
        return vec
    if len(vec) <= max_length:
        return vec
    if truncate_tail:
        return vec[:max_length]
    else:
        return vec[-max_length:]


def merge_utt(conversation, split_token_idx=None, keep_split_in_tail=False, final_token_idx=None):
    
    """
    from https://github.com/Zyh716/WSDM2022-C2CRS
    merge utterances in one conversation.

    Args:
        conversation (list of list of int): conversation consist of utterances consist of tokens.
        split_token_idx (int): index of split token. Defaults to None.
        keep_split_in_tail (bool): split in tail or head. Defaults to False.
        final_token_idx (int): index of final token. Defaults to None.

    Returns:
        list: tokens of all utterances in one list.

    """
    merged_conv = []
    for utt in conversation:
        for token in utt:
            merged_conv.append(token)
        if split_token_idx:
            merged_conv.append(split_token_idx)
    if split_token_idx and not keep_split_in_tail:
        merged_conv = merged_conv[:-1]
    if final_token_idx:
        merged_conv.append(final_token_idx)
    return merged_conv





def enumerateWithEstimate(
        iter,
        desc_str,
        start_ndx=0,
        print_ndx=4,
        backoff=None,
        iter_len=None,
):
    """
     from https://github.com/deep-learning-with-pytorch/dlwpt-code
    In terms of behavior, `enumerateWithEstimate` is almost identical
    to the standard `enumerate` (the differences are things like how
    our function returns a generator, while `enumerate` returns a
    specialized `<enumerate object at 0x...>`).

    However, the side effects (logging, specifically) are what make the
    function interesting.

    :param iter: `iter` is the iterable that will be passed into
        `enumerate`. Required.

    :param desc_str: This is a human-readable string that describes
        what the loop is doing. The value is arbitrary, but should be
        kept reasonably short. Things like `"epoch 4 training"` or
        `"deleting temp files"` or similar would all make sense.

    :param start_ndx: This parameter defines how many iterations of the
        loop should be skipped before timing actually starts. Skipping
        a few iterations can be useful if there are startup costs like
        caching that are only paid early on, resulting in a skewed
        average when those early iterations dominate the average time
        per iteration.

        NOTE: Using `start_ndx` to skip some iterations makes the time
        spent performing those iterations not be included in the
        displayed duration. Please account for this if you use the
        displayed duration for anything formal.

        This parameter defaults to `0`.

    :param print_ndx: determines which loop interation that the timing
        logging will start on. The intent is that we don't start
        logging until we've given the loop a few iterations to let the
        average time-per-iteration a chance to stablize a bit. We
        require that `print_ndx` not be less than `start_ndx` times
        `backoff`, since `start_ndx` greater than `0` implies that the
        early N iterations are unstable from a timing perspective.

        `print_ndx` defaults to `4`.

    :param backoff: This is used to how many iterations to skip before
        logging again. Frequent logging is less interesting later on,
        so by default we double the gap between logging messages each
        time after the first.

        `backoff` defaults to `2` unless iter_len is > 1000, in which
        case it defaults to `4`.

    :param iter_len: Since we need to know the number of items to
        estimate when the loop will finish, that can be provided by
        passing in a value for `iter_len`. If a value isn't provided,
        then it will be set by using the value of `len(iter)`.

    :return:
    """
    if iter_len is None:
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))
    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            # ... <1>
            duration_sec = ((time.time() - start_ts)
                            / (current_ndx - start_ndx + 1)
                            * (iter_len-start_ndx)
                            )

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            start_ts = time.time()

    log.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))

