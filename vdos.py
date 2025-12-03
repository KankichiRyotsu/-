"""
This module implements the functionality to calculate the phonon density of
states (PDOS) from MD trajectories. For parsing output files into a format
which is used here, see parse.py and test/* for examples. For a theory
overview, see [1]_.

Other codes which do that are [2]_ and [3]_.

.. [1] doc/source/written/background/phonon_dos.rst
.. [2] tfreq from Tim Teatro
       http://www.timteatro.net/2010/09/29/velocity-autocorrelation-and-vibrational-spectrum-calculation
.. [3] fourier.x from CPMD
"""

import os, warnings
import numpy as np
from numpy import s_
from scipy.fftpack import fft
from scipy.integrate import simpson as simps



def pyvacf(vel, m=None, method=3):
    """
    Reference implementation for calculating the VACF of velocities in 3d
    array `vel`. This is slow. Use for debugging only. For production, use
    fvacf().

    Parameters
    ----------
    vel : 3d array, (nstep, natoms, 3)
        Atomic velocities.
    m : 1d array (natoms,)
        Atomic masses.
    method : int
        | 1 : 3 loops
        | 2 : replace 1 inner loop
        | 3 : replace 2 inner loops

    Returns
    -------
    c : 1d array (nstep,)
        VACF
    """
    natoms = vel.shape[1]
    nstep = vel.shape[0]
    c = np.zeros((nstep,), dtype=float)
    if m is None:
        m = np.ones((natoms,), dtype=float)
    if method == 1:
        # c(t) = <v(t0) v(t0 + t)> / <v(t0)**2> = C(t) / C(0)
        #
        # "displacements" `t'
        for t in range(nstep):
            # time origins t0 == j
            for j in range(nstep-t):
                for i in range(natoms):
                    c[t] += np.dot(vel[j,i,:], vel[j+t,i,:]) * m[i]
    elif method == 2:
        # replace 1 inner loop
        for t in range(nstep):
            for j in range(nstep-t):
                # (natoms, 3) * (natoms, 1) -> (natoms, 3)
                c[t] += (vel[j,...] * vel[j+t,...] * m[:,None]).sum()
    elif method == 3:
        # replace 2 inner loops:
        # (xx, natoms, 3) * (1, natoms, 1) -> (xx, natoms, 3)
        for t in range(nstep):
            c[t] = (vel[:(nstep-t),...] * vel[t:,...]*m[None,:,None]).sum()
    else:
        raise ValueError('unknown method: %s' %method)
    # normalize to unity
    c = c / c[0]
    return c


def pad_zeros(arr, axis=0, where='end', nadd=None, upto=None, tonext=None,
              tonext_min=None):
    """
    Pad an nd-array with zeros. Default is to append an array of zeros of
    the same shape as `arr` to arr's end along `axis`.

    Parameters
    ----------
    arr :  nd array
    axis : the axis along which to pad
    where : string {'end', 'start'}, pad at the end ("append to array") or
        start ("prepend to array") of `axis`
    nadd : number of items to padd (i.e. nadd=3 means padd w/ 3 zeros in case
        of an 1d array)
    upto : pad until arr.shape[axis] == upto
    tonext : bool, pad up to the next power of two (pad so that the padded
        array has a length of power of two)
    tonext_min : int, when using `tonext`, pad the array to the next possible
        power of two for which the resulting array length along `axis` is at
        least `tonext_min`; the default is tonext_min = arr.shape[axis]

    Use only one of nadd, upto, tonext.

    Returns
    -------
    padded array

    Examples
    --------
    >>> # 1d
    >>> pad_zeros(a)
    array([1, 2, 3, 0, 0, 0])
    >>> pad_zeros(a, nadd=3)
    array([1, 2, 3, 0, 0, 0])
    >>> pad_zeros(a, upto=6)
    array([1, 2, 3, 0, 0, 0])
    >>> pad_zeros(a, nadd=1)
    array([1, 2, 3, 0])
    >>> pad_zeros(a, nadd=1, where='start')
    array([0, 1, 2, 3])
    >>> # 2d
    >>> a=arange(9).reshape(3,3)
    >>> pad_zeros(a, nadd=1, axis=0)
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8],
           [0, 0, 0]])
    >>> pad_zeros(a, nadd=1, axis=1)
    array([[0, 1, 2, 0],
           [3, 4, 5, 0],
           [6, 7, 8, 0]])
    >>> # up to next power of two
    >>> 2**arange(10)
    array([  1,   2,   4,   8,  16,  32,  64, 128, 256, 512])
    >>> pydos.pad_zeros(arange(9), tonext=True).shape
    (16,)
    """
    if tonext == False:
        tonext = None
    lst = [nadd, upto, tonext]
    assert lst.count(None) in [2,3], "`nadd`, `upto` and `tonext` must be " +\
           "all None or only one of them not None"
    if nadd is None:
        if upto is None:
            if (tonext is None) or (not tonext):
                # default
                nadd = arr.shape[axis]
            else:
                tonext_min = arr.shape[axis] if (tonext_min is None) \
                             else tonext_min
                # beware of int overflows starting w/ 2**arange(64), but we
                # will never have such long arrays anyway
                two_powers = 2**np.arange(30)
                assert tonext_min <= two_powers[-1], ("tonext_min exceeds max power of 2")
                power = two_powers[np.searchsorted(two_powers, tonext_min)]
                nadd = power - arr.shape[axis]
        else:
            nadd = upto - arr.shape[axis]
    if nadd == 0:
        return arr
    add_shape = list(arr.shape) # ex) [nstep, natoms, 3]
    add_shape[axis] = nadd # ex) [nadd, natoms, 3]
    add_shape = tuple(add_shape)
    if where == 'end':
        return np.concatenate((arr, np.zeros(add_shape, dtype=arr.dtype)), axis=axis)
        # [arr,
        # [0, 0, ... 0],    |
        # ...               } npadd
        # [0, 0, ... 0],]   |
    elif where == 'start':
        return np.concatenate((np.zeros(add_shape, dtype=arr.dtype), arr), axis=axis)
        # [[0, 0, ... 0],   |
        # ...               } npadd
        # [0, 0, ... 0],    |
        # arr]
    else:
        raise Exception("illegal `where` arg: %s" %where)


def welch(M, sym=1):
    """
    Welch window. Function skeleton shamelessly stolen from
    scipy.signal.windows.bartlett() and others.
    """
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1,dtype=float)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = np.arange(0,M)
    w = 1.0-((n-0.5*(M-1))/(0.5*(M-1)))**2.0
    if not sym and not odd:
        w = w[:-1]
    return w


def norm_int(y, x, area=1.0, scale=True, func=simps):
    """
    Normalize integral area of y(x) to `area`.

    Parameters
    ----------
    x,y : numpy 1d arrays
    area : float
    scale : bool, optional
        Scale x and y to the same order of magnitude before integration.
        This may be necessary to avoid numerical trouble if x and y have very
        different scales.
    func : callable
        Function to do integration (like scipy.integrate.{simps,trapz,...}
        Called as ``func(y,x=x)``. Default: simps

    Returns
    -------
    scaled y

    Notes
    -----
    The argument order y,x might be confusing. x,y would be more natural but we
    stick to the order used in the scipy.integrate routines.
    """
    if scale:
        fx = np.abs(x).max()
        fy = np.abs(y).max()
        sx = x / fx
        sy = y / fy
    else:
        fx = fy = 1.0
        sx, sy = x, y
    # Area under unscaled y(x).
    _area = func(sy, x=sx) * fx * fy
    return y * area / _area


def sum(arr, axis=None, keepdims=False, **kwds):
    """
    This numpy.sum() with some features implemented which can be found in
    numpy v1.7 and later: `axis` can be a tuple to select arbitrary axes to sum
    over.

    We also have a `keepdims` keyword, which however works completely different
    from numpy. Docstrings shamelessly stolen from numpy and adapted here
    and there.

    Parameters
    ----------
    arr : nd array
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed. The default (`axis` =
        `None`) is to perform a sum over all the dimensions of the input array.
        `axis` may be negative, in which case it counts from the last to the
        first axis.
        If this is a tuple of ints, a sum is performed on multiple
        axes, instead of a single axis or all the axes as before.
    keepdims : bool, optional
        If this is set to True, the axes from `axis` are left in the result
        and the reduction (sum) is performed for all remaining axes. Therefore,
        it reverses the `axis` to be summed over.
    **kwds : passed to np.sum().

    Examples
    --------
    >>> a=rand(2,3,4)
    >>> num.sum(a)
    12.073636268676152
    >>> a.sum()
    12.073636268676152
    >>> num.sum(a, axis=1).shape
    (2, 4)
    >>> num.sum(a, axis=(1,)).shape
    (2, 4)
    >>> # same as axis=1, i.e. it inverts the axis over which we sum
    >>> num.sum(a, axis=(0,2), keepdims=True).shape
    (2, 4)
    >>> # numpy's keepdims has another meaning: it leave the summed axis (0,2)
    >>> # as dimension of size 1 to allow broadcasting
    >>> numpy.sum(a, axis=(0,2), keepdims=True).shape
    (1, 3, 1)
    >>> num.sum(a, axis=(1,)) - num.sum(a, axis=1)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.]])
    >>> num.sum(a, axis=(0,2)).shape
    (3,)
    >>> num.sum(a, axis=(0,2)) - a.sum(axis=0).sum(axis=1)
    array([ 0.,  0.,  0.])
    """
    # Recursion rocks!
    def _sum(arr, tosum):
        if len(tosum) > 0:
            # Choose axis to sum over, remove from list w/ remaining axes.
            axis = tosum.pop(0)
            _arr = arr.sum(axis=axis)
            # arr has one dim less now. Rename remaining axes accordingly.
            _tosum = [xx - 1 if xx > axis else xx for xx in tosum]
            return _sum(_arr, _tosum)
        else:
            return arr

    axis_is_int = isinstance(axis, int) # axisが整数かどうか判定　True/Falseを返す
    if axis is None:
        if keepdims:
            raise Exception("axis=None + keepdims=True makes no sense")
        else:
            return np.sum(arr, axis=axis, **kwds)
    elif axis_is_int and not keepdims:
        return np.sum(arr, axis=axis, **kwds)
    else:
        if axis_is_int:
            tosum = [axis]
        elif isinstance(axis, tuple) or isinstance(axis, list):
            tosum = list(axis)
        else:
            raise Exception("illegal type for axis: %s" % str(type(axis)))
        if keepdims:
            alldims = range(arr.ndim)
            tosum = [xx for xx in alldims if xx not in tosum]
        return _sum(arr, tosum)


def slicetake(a, sl, axis=None, copy=False):
    """
    The equivalent of numpy.take(a, ..., axis=<axis>), but accepts slice
    objects instead of an index array. Also by default, it returns a *view* and
    no copy.

    Parameters
    ----------
    a : numpy ndarray
    sl : slice object, list or tuple of slice objects
        axis=<int>
            one slice object for *that* axis
        axis=None
            `sl` is a list or tuple of slice objects, one for each axis.
            It must index the whole array, i.e. len(sl) == len(a.shape).
    axis : {None, int}
    copy : bool, return a copy instead of a view

    Returns
    -------
    A view into `a` or copy of a slice of `a`.

    Examples
    --------
    >>> from numpy import s_
    >>> a = np.random.rand(20,20,20)
    >>> b1 = a[:,:,10:]
    >>> # single slice for axis 2
    >>> b2 = slicetake(a, s_[10:], axis=2)
    >>> # tuple of slice objects
    >>> b3 = slicetake(a, s_[:,:,10:])
    >>> (b2 == b1).all()
    True
    >>> (b3 == b1).all()
    True
    >>> # simple extraction too, sl = integer
    >>> (a[...,5] == slicetake(a, 5, axis=-1))
    True
    """
    # The long story
    # --------------
    #
    # 1) Why do we need that:
    #
    # # no problem
    # a[5:10:2]
    #
    # # the same, more general
    # sl = slice(5,10,2)
    # a[sl]
    #
    # But we want to:
    #  - Define (type in) a slice object only once.
    #  - Take the slice of different arrays along different axes.
    # Since numpy.take() and a.take() don't handle slice objects, one would
    # have to use direct slicing and pay attention to the shape of the array:
    #
    #     a[sl], b[:,:,sl,:], etc ...
    #
    # We want to use an 'axis' keyword instead. np.r_() generates index arrays
    # from slice objects (e.g r_[1:5] == r_[s_[1:5] ==r_[slice(1,5,None)]).
    # Since we need index arrays for numpy.take(), maybe we can use that? Like
    # so:
    #
    #     a.take(r_[sl], axis=0)
    #     b.take(r_[sl], axis=2)
    #
    # Here we have what we want: slice object + axis kwarg.
    # But r_[slice(...)] does not work for all slice types. E.g. not for
    #
    #     r_[s_[::5]] == r_[slice(None, None, 5)] == array([], dtype=int32)
    #     r_[::5]                                 == array([], dtype=int32)
    #     r_[s_[1:]]  == r_[slice(1, None, None)] == array([0])
    #     r_[1:]
    #         ValueError: dimensions too large.
    #
    # The returned index arrays are wrong (or we even get an exception).
    # The reason is given below.
    # Bottom line: We need this function.
    #
    # The reason for r_[slice(...)] gererating sometimes wrong index arrays is
    # that s_ translates a fancy index (1:, ::5, 1:10:2, ...) to a slice
    # object. This *always* works. But since take() accepts only index arrays,
    # we use r_[s_[<fancy_index>]], where r_ translates the slice object
    # prodced by s_ to an index array. THAT works only if start and stop of the
    # slice are known. r_ has no way of knowing the dimensions of the array to
    # be sliced and so it can't transform a slice object into a correct index
    # array in case of slice(<number>, None, None) or slice(None, None,
    # <number>).
    #
    # 2) Slice vs. copy
    #
    # numpy.take(a, array([0,1,2,3])) or a[array([0,1,2,3])] return a copy of
    # `a` b/c that's "fancy indexing". But a[slice(0,4,None)], which is the
    # same as indexing (slicing) a[:4], return *views*.

    if axis is None:
        slices = sl
    else:
        # Note that these are equivalent:
        #   a[:]
        #   a[s_[:]]
        #   a[slice(None)]
        #   a[slice(None, None, None)]
        #   a[slice(0, None, None)]
        slices = [slice(None)] * a.ndim
        slices[axis] = sl
    # a[...] can take a tuple or list of slice objects
    # a[x:y:z, i:j:k] is the same as
    # a[(slice(x,y,z), slice(i,j,k))] == a[[slice(x,y,z), slice(i,j,k)]]
    slices = tuple(slices)
    if copy:
        return a[slices].copy()
    else:
        return a[slices]


def mirror(arr, axis=0):
    """
    Mirror array `arr` at index 0 along `axis`.
    The length of the returned array is 2*arr.shape[axis]-1 .
    ex) 
    arr = [a0, a1, a2]
    arr[::-1]          -> [a2, a1, a0]
    arr[1:]            -> [a1, a2]
    mirror(arr)        -> [a2, a1, a0, a1, a2] 
    """
    return np.concatenate((arr[::-1],arr[1:]), axis=axis)


def pdos(vel, dt=1.0, m=None, full_out=False, area=1.0, window=True,
         npad=None, tonext=False, mirr=False, method='direct'):
    """
    Phonon DOS by FFT of the VACF or direct FFT of atomic velocities.

    Integral area is normalized to `area`. It is possible (and recommended) to
    zero-padd the velocities (see `npad`).

    Parameters
    ----------
    vel : 3d array (nstep, natoms, 3)
        atomic velocities
    dt : time step
    m : 1d array (natoms,),
        atomic mass array, if None then mass=1.0 for all atoms is used
    full_out : bool
    area : float
        normalize area under frequency-PDOS curve to this value
    window : bool
        use Welch windowing on data before FFT (reduces leaking effect,
        recommended)
    npad : {None, int}
        method='direct' only: Length of zero padding along `axis`. `npad=None`
        = no padding, `npad > 0` = pad by a length of ``(nstep-1)*npad``. `npad
        > 5` usually results in sufficient interpolation.
    tonext : bool
        method='direct' only: Pad `vel` with zeros along `axis` up to the next
        power of two after the array length determined by `npad`. This gives
        you speed, but variable (better) frequency resolution.
    mirr : bool
        method='vacf' only: mirror one-sided VACF at t=0 before fft

    Returns
    -------
    if full_out = False
        | ``(faxis, pdos)``
        | faxis : 1d array [1/unit(dt)]
        | pdos : 1d array, the phonon DOS, normalized to `area`
    if full_out = True
        | if method == 'direct':
        |     ``(faxis, pdos, (full_faxis, full_pdos, split_idx))``
        | if method == 'vavcf':
        |     ``(faxis, pdos, (full_faxis, full_pdos, split_idx, vacf, fft_vacf))``
        |     fft_vacf : 1d complex array, result of fft(vacf) or fft(mirror(vacf))
        |     vacf : 1d array, the VACF

    Examples
    --------
    >>> from pwtools.constants import fs,rcm_to_Hz
    >>> tr = Trajectory(...)
    >>> # freq in [Hz] if timestep in [s]
    >>> freq,dos = pdos(tr.velocity, m=tr.mass, dt=tr.timestep*fs,
    >>>                 method='direct', npad=1)
    >>> # frequency in [1/cm]
    >>> plot(freq/rcm_to_Hz, dos)

    Notes
    -----
    padding (only method='direct'): With `npad` we pad the velocities `vel`
    with ``npad*(nstep-1)`` zeros along `axis` (the time axis) before FFT
    b/c the signal is not periodic. For `npad=1`, this gives us the exact
    same spectrum and frequency resolution as with ``pdos(...,
    method='vacf',mirr=True)`` b/c the array to be fft'ed has length
    ``2*nstep-1`` along the time axis in both cases (remember that the
    array length = length of the time axis influences the freq.
    resolution). FFT is only fast for arrays with length = a power of two.
    Therefore, you may get very different fft speeds depending on whether
    ``2*nstep-1`` is a power of two or not (in most cases it won't). Try
    using `tonext` but remember that you get another (better) frequency
    resolution.

    References
    ----------
    [1] Phys Rev B 47(9) 4863, 1993

    See Also
    --------
    :func:`pwtools.signal.fftsample`
    :func:`pwtools.signal.acorr`
    :func:`direct_pdos`
    :func:`vacf_pdos`

    """
    mass = m
    # assume vel.shape = (nstep,natoms,3)
    axis = 0
    assert vel.shape[-1] == 3
    if mass is not None:
        assert len(mass) == vel.shape[1], "len(mass) != vel.shape[1]"
        # define here b/c may be used twice below
        mass_bc = mass[None,:,None]
        # -> mass_bs.shape = (1, natoms, 1)
    
    if window:
        sl = [None]*vel.ndim # [ None, None, None ]
        sl[axis] = slice(None)  # ':'
                                # sl = [ :, None, None ]
        vel2 = vel*(welch(vel.shape[axis])[tuple(sl)])
                        # welch(vel.shape[axis])[tuple(sl)] 
                        #     = welch(vel.shape[axis])[:, None, None]
    else:
        vel2 = vel
    
    # handle options which are mutually exclusive
    if method == 'vacf':
        assert npad in [0,None], "use npad={0,None} for method='vacf'"
    
    # padding
    if npad is not None:
        nadd = (vel2.shape[axis]-1)*npad
        if tonext:
            vel2 = pad_zeros(vel2, 
                             tonext=True,
                             tonext_min=vel2.shape[axis] + nadd,
                             axis=axis
                             )
        else:
            vel2 = pad_zeros(vel2, 
                             tonext=False, 
                             nadd=nadd, 
                             axis=axis
                             )
    
    if method == 'direct':
        full_fft_vel = np.abs(fft(vel2, axis=axis))**2.0
        full_faxis = np.fft.fftfreq(vel2.shape[axis], dt)
        split_idx = len(full_faxis)//2
        faxis = full_faxis[:split_idx]
        # First split the array, then multiply by `mass` and average. If
        # full_out, then we need full_fft_vel below, so copy before slicing.
        arr = full_fft_vel.copy() if full_out else full_fft_vel
        fft_vel = slicetake(arr, slice(0, split_idx), axis=axis, copy=False)
        if mass is not None:
            fft_vel *= mass_bc
        # average remaining axes, summing is enough b/c normalization is done below
        # sums: (nstep, natoms, 3) -> (nstep, natoms) -> (nstep,)
        pdos = sum(fft_vel, axis=axis, keepdims=True)
        default_out = (faxis, norm_int(pdos, faxis, area=area))
        if full_out:
            # have to re-calculate this here b/c we never calculate the full_pdos
            # normally
            if mass is not None:
                full_fft_vel *= mass_bc
            full_pdos = sum(full_fft_vel, axis=axis, keepdims=True)
            extra_out = (full_faxis, full_pdos, split_idx)
            return default_out + extra_out
        else:
            return default_out
    
    elif method == 'vacf':
        vacf = pyvacf(vel2, m=mass)
        if mirr:
            fft_vacf = fft(mirror(vacf))
        else:
            fft_vacf = fft(vacf)
        full_faxis = np.fft.fftfreq(fft_vacf.shape[axis], dt)
        full_pdos = np.abs(fft_vacf)
        split_idx = len(full_faxis)//2
        faxis = full_faxis[:split_idx]
        pdos = full_pdos[:split_idx]
        default_out = (faxis, norm_int(pdos, faxis, area=area))
        extra_out = (full_faxis, full_pdos, split_idx, vacf, fft_vacf)
        if full_out:
            return default_out + extra_out
        else:
            return default_out




