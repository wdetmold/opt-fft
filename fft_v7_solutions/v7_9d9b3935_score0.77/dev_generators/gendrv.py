#!/usr/bin/env python3
"""Emit implementation.c: codelets + per-size drivers."""
import sys
import gen

SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
BIG = {13, 17, 23, 36, 45, 64}
XSMAP = {6: 36, 8: 64, 13: 208, 17: 408, 23: 552, 36: 1448, 45: 2168, 64: 4616}
ZPMAP = {6: 6, 8: 8, 13: 16, 17: 24, 23: 24, 36: 40, 45: 48, 64: 72}


import re as _re
def specialize(text, oldname, newname, isv, osv):
    """Fix strides to compile-time constants and rename the function."""
    text = text.replace(f"static void {oldname}(", f"static __attribute__((noinline)) void {newname}(", 1)
    text = text.replace(f"void {oldname}(", f"void {newname}(", 1)
    text = text.replace("const double* ri, const double* ii, ptrdiff_t is, double* ro, double* io, ptrdiff_t os",
                        "const double* ri, const double* ii, double* ro, double* io")
    text = _re.sub(r"(\d+)\*is", lambda m: str(int(m.group(1)) * isv), text)
    text = _re.sub(r"(\d+)\*os", lambda m: str(int(m.group(1)) * osv), text)
    assert "*is" not in text and "*os" not in text, newname
    return text

def chunks_cover(n, w):
    """chunk starts of width w covering [0,n) with tail overlap; starts+w<=n"""
    if n == w:
        return [0]
    if n < w:
        return None  # cannot cover without spill
    out = list(range(0, n - w + 1, w))
    if out[-1] + w < n:
        out.append(n - w)
    return out

def ychunks(L):
    # cover [0,L) with w8 (fallback w4 chunks)
    if L >= 8:
        return [(s, 8) for s in chunks_cover(L, 8)]
    return [(s, 4) for s in chunks_cover(L, 4)]

HEADER = r'''
// ============================================================================
// Iterated batched 3D complex FFT pipeline for L in {6,8,13,17,23,36,45,64}.
//
// All transform arithmetic in this file is original, generated at development
// time by my own codelet generator (mixed-radix Cooley-Tukey, prime-factor
// (Good-Thomas), Rader prime-length FFTs, and folded direct DFTs), emitted as
// straight-line AVX-512 code over split real/imaginary planes. No FFT library
// code, binaries, or tables are used anywhere.
//
// Per step (z = FFT3(x) + c; x <- z/(1+|z|)):
//   - pass over y (vertical SIMD, lanes across contiguous z),
//   - pass over z via in-register 8x8 transposes (swaps the inner layout),
//   - pass over x, blocked into two in-L2 stages for 36/45/64, fused with the
//     +c and the magnitude map (Newton iterations from rsqrt14/rcp14 seeds,
//     full double accuracy).
// Between steps the volume layout alternates orientation; c is kept in the
// matching layouts. The x-blocked sizes keep their planes digit-reversed with
// a self-inverse permutation so everything stays in place.
//
// Input generation reproduces numpy's Generator(PCG64(seed)).standard_normal
// bit-exactly (PCG64 with 4-way jump-ahead + the ziggurat tables extracted
// from this machine's numpy, validated element-exact at import; falls back to
// numpy generation if the check ever fails).
// Auto-generated. Own-arithmetic FFT implementation (no FFT libraries).
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <sys/mman.h>

typedef double V8 __attribute__((vector_size(64), aligned(8)));
typedef double V4 __attribute__((vector_size(32), aligned(8)));
typedef double V2 __attribute__((vector_size(16), aligned(8)));
#define K8(x) ((V8){(x),(x),(x),(x),(x),(x),(x),(x)})
#define K4(x) ((V4){(x),(x),(x),(x)})
#define K2(x) ((V2){(x),(x)})
#define K1(x) (x)
#define LD1(p) (*(p))
#define ST1(p,v) (*(p) = (v))
#define LD8(p) (*(const V8*)(p))
#define LD4(p) (*(const V4*)(p))
#define LD2(p) (*(const V2*)(p))
#define ST8(p,v) (*(V8*)(p) = (v))
#define ST4(p,v) (*(V4*)(p) = (v))
#define ST2(p,v) (*(V2*)(p) = (v))


// ---------------- bit-exact PCG64 + ziggurat standard normal ----------------
#include <math.h>
// ziggurat tables (bit patterns, extracted at dev time; validated at import)
static const uint64_t ki_double[256] = {0xef33d8025ef6aULL,0x0ULL,0xc08be98fbc6a8ULL,0xda354fabd8142ULL,0xe51f67ec1eeeaULL,0xeb255e9d3f77eULL,0xeef4b817ecab9ULL,0xf19470afa44aaULL,0xf37ed61ffcb18ULL,0xf4f469561255cULL,0xf61a5e41ba396ULL,0xf707a755396a4ULL,0xf7cb2ec28449aULL,0xf86f10c6357d3ULL,0xf8fa6578325deULL,0xf9724c74dd0daULL,0xf9da907dbf509ULL,0xfa360f581fa74ULL,0xfa86fde5b4bf8ULL,0xfacf160d354dcULL,0xfb0fb6718b90fULL,0xfb49f8d5374c6ULL,0xfb7ec2366fe77ULL,0xfbaece9a1e50eULL,0xfbdab9d040bedULL,0xfc03060ff6c57ULL,0xfc2821037a248ULL,0xfc4a67ae25bd1ULL,0xfc6a2977aee31ULL,0xfc87aa92896a4ULL,0xfca325e4bde85ULL,0xfcbcce902231aULL,0xfcd4d12f839c4ULL,0xfceb54d8fec99ULL,0xfd007bf1dc930ULL,0xfd1464dd6c4e6ULL,0xfd272a8e2f450ULL,0xfd38e4ff0c91eULL,0xfd49a9990b478ULL,0xfd598b8920f53ULL,0xfd689c08e99ecULL,0xfd76ea9c8e832ULL,0xfd848547b08e8ULL,0xfd9178bad2c8cULL,0xfd9dd07a7add2ULL,0xfda9970105e8cULL,0xfdb4d5dc02e20ULL,0xfdbf95c5bfcd0ULL,0xfdc9debb99a7dULL,0xfdd3b8118729dULL,0xfddd288342f90ULL,0xfde6364369f64ULL,0xfdeee708d514eULL,0xfdf7401a6b42eULL,0xfdff46599ed40ULL,0xfe06fe4bc24f2ULL,0xfe0e6c225a258ULL,0xfe1593c28b84cULL,0xfe1c78cbc3f99ULL,0xfe231e9db1caaULL,0xfe29885da1b91ULL,0xfe2fb8fb54186ULL,0xfe35b33558d4aULL,0xfe3b799d0002aULL,0xfe410e99ead7fULL,0xfe46746d47734ULL,0xfe4bad34c095cULL,0xfe50baed29524ULL,0xfe559f74ebc78ULL,0xfe5a5c8e41212ULL,0xfe5ef3e138689ULL,0xfe6366fd91078ULL,0xfe67b75c6d578ULL,0xfe6be661e11aaULL,0xfe6ff55e5f4f2ULL,0xfe73e5900a702ULL,0xfe77b823e9e39ULL,0xfe7b6e37070a2ULL,0xfe7f08d774243ULL,0xfe8289053f08cULL,0xfe85efb35173aULL,0xfe893dc840864ULL,0xfe8c741f0cebcULL,0xfe8f9387d4ef6ULL,0xfe929cc879b1dULL,0xfe95909d388eaULL,0xfe986fb939aa2ULL,0xfe9b3ac714866ULL,0xfe9df2694b6d5ULL,0xfea0973abe67cULL,0xfea329cf166a4ULL,0xfea5aab32952cULL,0xfea81a6d5741aULL,0xfeaa797de1cf0ULL,0xfeacc85f3d920ULL,0xfeaf07865e63cULL,0xfeb13762fec13ULL,0xfeb3585fe2a4aULL,0xfeb56ae3162b4ULL,0xfeb76f4e284faULL,0xfeb965fe62014ULL,0xfebb4f4cf9d7cULL,0xfebd2b8f449d0ULL,0xfebefb16e2e3eULL,0xfec0be31ebde8ULL,0xfec2752b15a15ULL,0xfec42049dafd3ULL,0xfec5bfd29f196ULL,0xfec75406ceef4ULL,0xfec8dd2500cb4ULL,0xfeca5b6911f12ULL,0xfecbcf0c427feULL,0xfecd38454fb15ULL,0xfece97488c8b3ULL,0xfecfec47f91b7ULL,0xfed1377358528ULL,0xfed278f844903ULL,0xfed3b10242f4cULL,0xfed4dfbad586eULL,0xfed605498c3ddULL,0xfed721d414fe8ULL,0xfed8357e4a982ULL,0xfed9406a42cc8ULL,0xfeda42b85b704ULL,0xfedb3c8746ab4ULL,0xfedc2df416652ULL,0xfedd171a46e52ULL,0xfeddf813c8ad3ULL,0xfeded0f909980ULL,0xfedfa1e0fd414ULL,0xfee06ae124bc4ULL,0xfee12c0d95a06ULL,0xfee1e579006e0ULL,0xfee29734b6524ULL,0xfee34150ae4bcULL,0xfee3e3db89b3cULL,0xfee47ee2982f4ULL,0xfee51271db086ULL,0xfee59e9407f41ULL,0xfee623528b42eULL,0xfee6a0b5897f1ULL,0xfee716c3e077aULL,0xfee7858327b82ULL,0xfee7ecf7b06baULL,0xfee84d2484ab2ULL,0xfee8a60b66343ULL,0xfee8f7accc851ULL,0xfee94207e25daULL,0xfee9851a829eaULL,0xfee9c0e13485cULL,0xfee9f557273f4ULL,0xfeea22762ccaeULL,0xfeea4836b42acULL,0xfeea668fc2d71ULL,0xfeea7d76ed6faULL,0xfeea8ce04fa0aULL,0xfeea94be8333bULL,0xfeea950296410ULL,0xfeea8d9c0075eULL,0xfeea7e7897654ULL,0xfeea678481d24ULL,0xfeea48aa29e83ULL,0xfeea21d22e4daULL,0xfee9f2e352024ULL,0xfee9bbc26af2eULL,0xfee97c524f2e4ULL,0xfee93473c0a3aULL,0xfee8e40557516ULL,0xfee88ae369c7aULL,0xfee828e7f3dfdULL,0xfee7bdea7b888ULL,0xfee749bff37ffULL,0xfee6cc3a9bd5eULL,0xfee64529e007eULL,0xfee5b45a32888ULL,0xfee51994e57b6ULL,0xfee474a0006cfULL,0xfee3c53e12c50ULL,0xfee30b2e02ad8ULL,0xfee2462ad8205ULL,0xfee175eb83c5aULL,0xfee09a22a1447ULL,0xfedfb27e349ccULL,0xfedebea76216cULL,0xfeddbe422047eULL,0xfedcb0ece39d3ULL,0xfedb964042cf4ULL,0xfeda6dce938c9ULL,0xfed937237e98dULL,0xfed7f1c38a836ULL,0xfed69d2b9c02bULL,0xfed538d06ae00ULL,0xfed3c41dea422ULL,0xfed23e76a2fd8ULL,0xfed0a732fe644ULL,0xfecefda07fe34ULL,0xfecd4100eb7b8ULL,0xfecb708956eb4ULL,0xfec98b61230c1ULL,0xfec790a0da978ULL,0xfec57f50f31feULL,0xfec356686c962ULL,0xfec114cb4b335ULL,0xfebeb948e6fd0ULL,0xfebc429a0b692ULL,0xfeb9af5ee0cdcULL,0xfeb6fe1c98542ULL,0xfeb42d3ad1f9eULL,0xfeb13b00b2d4bULL,0xfeae2591a02e9ULL,0xfeaaeae992257ULL,0xfea788d8ee326ULL,0xfea3fcffd73e5ULL,0xfea044c8dd9f6ULL,0xfe9c5d62f563bULL,0xfe9843ba947a4ULL,0xfe93f471d4728ULL,0xfe8f6bd76c5d6ULL,0xfe8aa5dc4e8e6ULL,0xfe859e07ab1eaULL,0xfe804f690a940ULL,0xfe7ab488233c0ULL,0xfe74c751f6aa5ULL,0xfe6e8102aa202ULL,0xfe67da0b6abd8ULL,0xfe60c9f38307eULL,0xfe5947338f742ULL,0xfe51470977280ULL,0xfe48bd436f458ULL,0xfe3f9bffd1e37ULL,0xfe35d35eeb19cULL,0xfe2b5122fe4feULL,0xfe20003995557ULL,0xfe13c82788314ULL,0xfe068c4ee67b0ULL,0xfdf82b02b71aaULL,0xfde87c57efeaaULL,0xfdd7509c63bfdULL,0xfdc46e529bf13ULL,0xfdaf8f82e0282ULL,0xfd985e1b2ba75ULL,0xfd7e6ef48cf04ULL,0xfd613adbd650bULL,0xfd40149e2f012ULL,0xfd1a1a7b4c7acULL,0xfcee204761f9eULL,0xfcba8d85e11b2ULL,0xfc7d26ecd2d22ULL,0xfc32b2f1e22edULL,0xfbd6581c0b83aULL,0xfb606c4005434ULL,0xfac40582a2874ULL,0xf9e971e014598ULL,0xf89fa48a41dfcULL,0xf66c5f7f0302cULL,0xf1a5a4b331c4aULL};
static const uint64_t wi_double_bits[256] = {0x3ccf493b7815d979ULL,0x3c8b8d0be3fdf6c6ULL,0x3c9250af3c2c5bb4ULL,0x3c957cb938443b61ULL,0x3c9801fce82fa70cULL,0x3c9a230c2e4cd0bcULL,0x3c9c004d2f3861f7ULL,0x3c9dac2f5a747274ULL,0x3c9f32482d4cd5c3ULL,0x3ca04d32278ebbadULL,0x3ca0f5053b025d43ULL,0x3ca192a697413677ULL,0x3ca227a28f7a1af5ULL,0x3ca2b52e3863d880ULL,0x3ca33c3fc05791f5ULL,0x3ca3bd9ec1a2b12fULL,0x3ca439ef8dff9b55ULL,0x3ca4b1bb363dfea7ULL,0x3ca52575621ad374ULL,0x3ca59580a707ce96ULL,0x3ca60231cfd97eeaULL,0x3ca66bd261a37c3dULL,0x3ca6d2a292000570ULL,0x3ca736dad346f8a6ULL,0x3ca798ad10b32a77ULL,0x3ca7f845ad46f543ULL,0x3ca855cc53430a77ULL,0x3ca8b1649e7b769aULL,0x3ca90b2ea94ecf98ULL,0x3ca96347822c1eeaULL,0x3ca9b9c98e38c546ULL,0x3caa0eccdca4a72cULL,0x3caa62676d77cd59ULL,0x3caab4ad6e101630ULL,0x3cab05b16d136c9cULL,0x3cab558487427a29ULL,0x3caba4368e529f3aULL,0x3cabf1d62abf8232ULL,0x3cac3e70f9594ef3ULL,0x3cac8a13a5323b61ULL,0x3cacd4c9fe72268bULL,0x3cad1e9f0e80b748ULL,0x3cad679d29e41f10ULL,0x3cadafce0023b8c3ULL,0x3cadf73aa9f17653ULL,0x3cae3debb5d2edfeULL,0x3cae83e9337a6f00ULL,0x3caec93abdf982ceULL,0x3caf0de784f06226ULL,0x3caf51f654d8f688ULL,0x3caf956d9e87d7aeULL,0x3cafd8537dfa2eacULL,0x3cb00d56e04234ecULL,0x3cb02e40f5398f9aULL,0x3cb04eea9e16a5fcULL,0x3cb06f565b72a010ULL,0x3cb08f869071f40bULL,0x3cb0af7d84bc6113ULL,0x3cb0cf3d664bcc7fULL,0x3cb0eec84b16086bULL,0x3cb10e20329515eeULL,0x3cb12d4707310fbeULL,0x3cb14c3e9f8e9141ULL,0x3cb16b08bfc4201eULL,0x3cb189a71a78da34ULL,0x3cb1a81b51ee6d88ULL,0x3cb1c666f8f82acbULL,0x3cb1e48b93e0d42eULL,0x3cb2028a9940a09fULL,0x3cb2206572c4c6e9ULL,0x3cb23e1d7de9c31fULL,0x3cb25bb40ca96bfbULL,0x3cb2792a661dd37fULL,0x3cb29681c719d71bULL,0x3cb2b3bb62b82edaULL,0x3cb2d0d862e1b853ULL,0x3cb2edd9e8cba98eULL,0x3cb30ac10d6e48d7ULL,0x3cb3278ee1f4b930ULL,0x3cb3444470265ea1ULL,0x3cb360e2baca52d5ULL,0x3cb37d6abe05586aULL,0x3cb399dd6fb2b264ULL,0x3cb3b63bbfb83d03ULL,0x3cb3d28698561de0ULL,0x3cb3eebede725a83ULL,0x3cb40ae571e09e74ULL,0x3cb426fb2da6745dULL,0x3cb44300e83c30a4ULL,0x3cb45ef773cac75dULL,0x3cb47adf9e66c336ULL,0x3cb496ba32488f2fULL,0x3cb4b287f602415dULL,0x3cb4ce49acb311dcULL,0x3cb4ea001638a605ULL,0x3cb505abef5e5562ULL,0x3cb5214df20a8b5aULL,0x3cb53ce6d56a664fULL,0x3cb558774e1bb2c8ULL,0x3cb574000e555f78ULL,0x3cb58f81c60e8514ULL,0x3cb5aafd23241b59ULL,0x3cb5c672d17d733dULL,0x3cb5e1e37b2f8cd3ULL,0x3cb5fd4fc89f5e38ULL,0x3cb618b860a31fc3ULL,0x3cb6341de8a2b0a2ULL,0x3cb64f8104b7260bULL,0x3cb66ae257c99672ULL,0x3cb6864283b13137ULL,0x3cb6a1a22950b2b1ULL,0x3cb6bd01e8b343bbULL,0x3cb6d8626128d352ULL,0x3cb6f3c43161f854ULL,0x3cb70f27f78b68ebULL,0x3cb72a8e516914c6ULL,0x3cb745f7dc70eedcULL,0x3cb7616535e5731fULL,0x3cb77cd6faeff449ULL,0x3cb7984dc8babd93ULL,0x3cb7b3ca3c8b1409ULL,0x3cb7cf4cf3db22fbULL,0x3cb7ead68c73dee7ULL,0x3cb80667a486ea1fULL,0x3cb82200dac88676ULL,0x3cb83da2ce899f15ULL,0x3cb8594e1fd1f5bdULL,0x3cb875036f7a7ec5ULL,0x3cb890c35f47f72dULL,0x3cb8ac8e9205c043ULL,0x3cb8c865aba10c9cULL,0x3cb8e44951446a27ULL,0x3cb9003a2973b58fULL,0x3cb91c38dc288347ULL,0x3cb9384612ef0afcULL,0x3cb954627903a28aULL,0x3cb9708ebb70d5eeULL,0x3cb98ccb892e2a31ULL,0x3cb9a919933f99bfULL,0x3cb9c5798cd5d92cULL,0x3cb9e1ec2b6f7411ULL,0x3cb9fe7226fad24aULL,0x3cba1b0c39f93692ULL,0x3cba37bb21a2c85bULL,0x3cba547f9e0bbb88ULL,0x3cba715a724aa9a4ULL,0x3cba8e4c64a0313dULL,0x3cbaab563e9ff108ULL,0x3cbac878cd5af5ceULL,0x3cbae5b4e18bb336ULL,0x3cbb030b4fc3a11aULL,0x3cbb207cf09a985bULL,0x3cbb3e0aa0e00c00ULL,0x3cbb5bb541ce3d03ULL,0x3cbb797db93f8927ULL,0x3cbb9764f1e5f73cULL,0x3cbbb56bdb85256eULL,0x3cbbd3936b2ec0a2ULL,0x3cbbf1dc9b81ae83ULL,0x3cbc10486cec16a0ULL,0x3cbc2ed7e5f07a2dULL,0x3cbc4d8c136e0d1cULL,0x3cbc6c6608ec8705ULL,0x3cbc8b66e0eba617ULL,0x3cbcaa8fbd36a2abULL,0x3cbcc9e1c73bd690ULL,0x3cbce95e3068e037ULL,0x3cbd0906328b8f6eULL,0x3cbd28db1037ef20ULL,0x3cbd48de1533c647ULL,0x3cbd691096e7f123ULL,0x3cbd8973f4d7fba5ULL,0x3cbdaa0999206e70ULL,0x3cbdcad2f8fc490eULL,0x3cbdebd195522e37ULL,0x3cbe0d06fb49d21cULL,0x3cbe2e74c4ea46f6ULL,0x3cbe501c99c1d188ULL,0x3cbe72002f97fe25ULL,0x3cbe94214b2abf0aULL,0x3cbeb681c0f76f08ULL,0x3cbed9237610a73aULL,0x3cbefc086101eca9ULL,0x3cbf1f328ac25321ULL,0x3cbf42a40fb74d6dULL,0x3cbf665f20c90168ULL,0x3cbf8a6604899782ULL,0x3cbfaebb187122bfULL,0x3cbfd360d22fe785ULL,0x3cbff859c118f60bULL,0x3cc00ed447d3a075ULL,0x3cc021a8028fc947ULL,0x3cc034a983a902abULL,0x3cc047da4e3ef5c7ULL,0x3cc05b3bf6adb37eULL,0x3cc06ed023a72668ULL,0x3cc082988f632e17ULL,0x3cc0969708e8a254ULL,0x3cc0aacd7571c0c4ULL,0x3cc0bf3dd1eed448ULL,0x3cc0d3ea34aa3d30ULL,0x3cc0e8d4cf116593ULL,0x3cc0fdffefa69fb6ULL,0x3cc1136e04207041ULL,0x3cc129219bbb5d35ULL,0x3cc13f1d69c4096dULL,0x3cc1556448602e3bULL,0x3cc16bf93b9deef3ULL,0x3cc182df74d21261ULL,0x3cc19a1a564eebacULL,0x3cc1b1ad777f2f8eULL,0x3cc1c99ca971a694ULL,0x3cc1e1ebfbe4ae39ULL,0x3cc1fa9fc2e2d901ULL,0x3cc213bc9d04cc81ULL,0x3cc22d477a6fd3eeULL,0x3cc24745a4ac9c24ULL,0x3cc261bcc77658e0ULL,0x3cc27cb2faa8592eULL,0x3cc2982ecd770e78ULL,0x3cc2b437532a0a52ULL,0x3cc2d0d43196db97ULL,0x3cc2ee0db1a978f5ULL,0x3cc30becd256aeeeULL,0x3cc32a7b5e68a4a3ULL,0x3cc349c405ae12a3ULL,0x3cc369d27a33a840ULL,0x3cc38ab39256410aULL,0x3cc3ac7570ae88faULL,0x3cc3cf27b31704a6ULL,0x3cc3f2dbaa60f475ULL,0x3cc417a49cb9e5daULL,0x3cc43d9815545e94ULL,0x3cc464ce44a73a15ULL,0x3cc48d62759c43bcULL,0x3cc4b7739d6b5a27ULL,0x3cc4e3250dcd8902ULL,0x3cc5109f53e9ac41ULL,0x3cc54011523a7e42ULL,0x3cc571b1a94ae41bULL,0x3cc5a5c08b718dd9ULL,0x3cc5dc8a243ad0feULL,0x3cc61669cf861e4cULL,0x3cc653ce7b006aeaULL,0x3cc69540be9fe5c3ULL,0x3cc6db6b8d09e232ULL,0x3cc72728f05f7a34ULL,0x3cc7799556090673ULL,0x3cc7d42df4d6ce8cULL,0x3cc839030529f234ULL,0x3cc8ab0fbfaa7c14ULL,0x3cc92ee0946f4496ULL,0x3cc9cbee014057abULL,0x3cca8fdc7894775aULL,0x3ccb981f3878fdb1ULL,0x3ccd3bb48209ad33ULL};
static const uint64_t fi_double_bits[256] = {0x3ff0000000000000ULL,0x3fef446ac979f087ULL,0x3feeb7545b6ca915ULL,0x3fee3f11e027f077ULL,0x3fedd36fa704de95ULL,0x3fed70920657bcf2ULL,0x3fed144978a119dcULL,0x3fecbd33a8a72debULL,0x3fec6a5ecea9787fULL,0x3fec1b1cd9eebaeaULL,0x3febceeb4ee1dc82ULL,0x3feb85653a8ff552ULL,0x3feb3e3a8234dd10ULL,0x3feaf92a3f6ce8a2ULL,0x3feab5fef17a2504ULL,0x3fea748bd550c9e1ULL,0x3fea34aafdf5af0fULL,0x3fe9f63bee651fd8ULL,0x3fe9b9228d240681ULL,0x3fe97d4657617ac1ULL,0x3fe94291c21b7a47ULL,0x3fe908f1bd31714fULL,0x3fe8d0554fe60aa8ULL,0x3fe898ad48badf02ULL,0x3fe861ebfc37bcacULL,0x3fe82c050f56cf6eULL,0x3fe7f6ed4b20e2cbULL,0x3fe7c29a779c6858ULL,0x3fe78f033ca0b0d5ULL,0x3fe75c1f0770d856ULL,0x3fe729e5f43f6d12ULL,0x3fe6f850baea7aeeULL,0x3fe6c7589e635a89ULL,0x3fe696f75e513b2aULL,0x3fe667272a92e323ULL,0x3fe637e298550c18ULL,0x3fe6092498802665ULL,0x3fe5dae86f4aff6aULL,0x3fe5ad29acc85c89ULL,0x3fe57fe4264c8d8fULL,0x3fe55313f08d9e46ULL,0x3fe526b55a656cd5ULL,0x3fe4fac4e820b667ULL,0x3fe4cf3f4f494ec0ULL,0x3fe4a42172dc5278ULL,0x3fe479685fdf5012ULL,0x3fe44f114a493679ULL,0x3fe425198a355fe3ULL,0x3fe3fb7e99585b82ULL,0x3fe3d23e10af31a3ULL,0x3fe3a955a662cd0eULL,0x3fe380c32bda00d5ULL,0x3fe358848bf550e9ULL,0x3fe33097c9703a35ULL,0x3fe308fafd6438efULL,0x3fe2e1ac55ea3beeULL,0x3fe2baaa14d7954aULL,0x3fe293f28e93cd15ULL,0x3fe26d84290504edULL,0x3fe2475d5a90db84ULL,0x3fe2217ca92ff7f2ULL,0x3fe1fbe0a9929620ULL,0x3fe1d687fe549969ULL,0x3fe1b171573fd111ULL,0x3fe18c9b709b3c50ULL,0x3fe16805128639daULL,0x3fe143ad105ea99cULL,0x3fe11f9248311f38ULL,0x3fe0fbb3a2325913ULL,0x3fe0d810104142a0ULL,0x3fe0b4a68d70d9aeULL,0x3fe091761d995d81ULL,0x3fe06e7dccf03c36ULL,0x3fe04bbcafa63f2eULL,0x3fe02931e18b822aULL,0x3fe006dc85b8cac4ULL,0x3fdfc9778c7bbda1ULL,0x3fdf859da7a900caULL,0x3fdf4229cb2f7af3ULL,0x3fdeff1a717e8f95ULL,0x3fdebc6e20bd1f54ULL,0x3fde7a236a4ec3c5ULL,0x3fde3838ea5f9b85ULL,0x3fddf6ad47763a09ULL,0x3fddb57f320b56b1ULL,0x3fdd74ad6426de33ULL,0x3fdd3436a1021080ULL,0x3fdcf419b4ae5b6dULL,0x3fdcb45573c0a848ULL,0x3fdc74e8bb00d7c7ULL,0x3fdc35d26f1d2cb8ULL,0x3fdbf7117c616a17ULL,0x3fdbb8a4d6716d91ULL,0x3fdb7a8b7807131bULL,0x3fdb3cc462b331caULL,0x3fdaff4e9ea18552ULL,0x3fdac2293a5f5a9eULL,0x3fda85534aa4d880ULL,0x3fda48cbea20c04dULL,0x3fda0c923946843eULL,0x3fd9d0a55e1e93dfULL,0x3fd995048418c0c6ULL,0x3fd959aedbe09f93ULL,0x3fd91ea39b33cb17ULL,0x3fd8e3e1fcb9f115ULL,0x3fd8a9693fde9188ULL,0x3fd86f38a8ac5ab6ULL,0x3fd8354f7faa0dd9ULL,0x3fd7fbad11b8d911ULL,0x3fd7c250aff414b0ULL,0x3fd78939af9252ebULL,0x3fd7506769c7b1edULL,0x3fd717d93ba9614cULL,0x3fd6df8e86124caaULL,0x3fd6a786ad88de21ULL,0x3fd66fc11a25cbe2ULL,0x3fd6383d377be515ULL,0x3fd600fa7480d2c8ULL,0x3fd5c9f84376c244ULL,0x3fd5933619d6eebeULL,0x3fd55cb3703d0100ULL,0x3fd5266fc2533bedULL,0x3fd4f06a8ebf6d92ULL,0x3fd4baa357109ca2ULL,0x3fd485199fad6ad4ULL,0x3fd44fccefc324feULL,0x3fd41abcd1357a19ULL,0x3fd3e5e8d08ed2dbULL,0x3fd3b1507cf143aeULL,0x3fd37cf368081379ULL,0x3fd348d125f9d19eULL,0x3fd314e94d5af62fULL,0x3fd2e13b77210766ULL,0x3fd2adc73e963fddULL,0x3fd27a8c414db11eULL,0x3fd2478a1f17de89ULL,0x3fd214c079f7cc9eULL,0x3fd1e22ef6188116ULL,0x3fd1afd539c2f050ULL,0x3fd17db2ed5454e8ULL,0x3fd14bc7bb34ee67ULL,0x3fd11a134fcf2423ULL,0x3fd0e895598709c4ULL,0x3fd0b74d88b242daULL,0x3fd0863b8f904336ULL,0x3fd0555f2242e9d9ULL,0x3fd024b7f6c7747eULL,0x3fcfe88b89df93c5ULL,0x3fcf88108cb83235ULL,0x3fcf27fe6ce998d2ULL,0x3fcec854a4c99c44ULL,0x3fce6912b2283cddULL,0x3fce0a3816457184ULL,0x3fcdabc455c7900aULL,0x3fcd4db6f8b2514fULL,0x3fccf00f8a5e6fccULL,0x3fcc92cd9971df53ULL,0x3fcc35f0b7d89d47ULL,0x3fcbd9787abe18a1ULL,0x3fcb7d647a8731aaULL,0x3fcb21b452ccd13aULL,0x3fcac667a2571807ULL,0x3fca6b7e0b19267eULL,0x3fca10f7322d7e3dULL,0x3fc9b6d2bfd2fe5aULL,0x3fc95d105f6a7c27ULL,0x3fc903afbf74fa69ULL,0x3fc8aab09192815bULL,0x3fc852128a819a38ULL,0x3fc7f9d5621f7175ULL,0x3fc7a1f8d368a323ULL,0x3fc74a7c9c7ab5a6ULL,0x3fc6f3607e964716ULL,0x3fc69ca43e21f25cULL,0x3fc64647a2adf19cULL,0x3fc5f04a76f883f9ULL,0x3fc59aac88f31d6cULL,0x3fc5456da9c86835ULL,0x3fc4f08dade31fc1ULL,0x3fc49c0c6cf5ce2dULL,0x3fc447e9c20375d5ULL,0x3fc3f4258b6931aeULL,0x3fc3a0bfaae8d7eeULL,0x3fc34db805b4ab88ULL,0x3fc2fb0e847c2a65ULL,0x3fc2a8c3137a071aULL,0x3fc256d5a2835eb7ULL,0x3fc2054625183c34ULL,0x3fc1b41492757d42ULL,0x3fc16340e5a82d63ULL,0x3fc112cb1da26eb9ULL,0x3fc0c2b33d5209baULL,0x3fc072f94bb8bf85ULL,0x3fc0239d54067d2aULL,0x3fbfa93ecb6b222cULL,0x3fbf0bff29520e1cULL,0x3fbe6f7bf29aa54bULL,0x3fbdd3b56176e88fULL,0x3fbd38abb9bd91e5ULL,0x3fbc9e5f493b740aULL,0x3fbc04d0680b1015ULL,0x3fbb6bff78f2e233ULL,0x3fbad3ece9caf633ULL,0x3fba3c9933ea6286ULL,0x3fb9a604dc9d5b19ULL,0x3fb9103075a4a0abULL,0x3fb87b1c9dbf2852ULL,0x3fb7e6ca013eefd6ULL,0x3fb753395aaa1176ULL,0x3fb6c06b73694a4cULL,0x3fb62e6124854d18ULL,0x3fb59d1b577466a4ULL,0x3fb50c9b06fa2baeULL,0x3fb47ce1401b2213ULL,0x3fb3edef23269a86ULL,0x3fb35fc5e4d93e70ULL,0x3fb2d266cf9b3111ULL,0x3fb245d344dd0d91ULL,0x3fb1ba0cbe97897dULL,0x3fb12f14d0f2179dULL,0x3fb0a4ed2c159625ULL,0x3fb01b979e30e497ULL,0x3faf262c2b6c6e35ULL,0x3fae16d547b25181ULL,0x3fad092efeadf162ULL,0x3fabfd3e0f282a2cULL,0x3faaf30790385f70ULL,0x3fa9ea90f9295563ULL,0x3fa8e3e02a68b5abULL,0x3fa7defb77af271eULL,0x3fa6dbe9b398d064ULL,0x3fa5dab23cf2add4ULL,0x3fa4db5d0e11275dULL,0x3fa3ddf2ce98eecbULL,0x3fa2e27ce83df497ULL,0x3fa1e9059f1f6abcULL,0x3fa0f1982e968011ULL,0x3f9ff881d718a5c4ULL,0x3f9e121adb828c75ULL,0x3f9c301983cd091aULL,0x3f9a529f4e22ebf8ULL,0x3f9879d1b600c10aULL,0x3f96a5daf40bbf82ULL,0x3f94d6eaf2fbb064ULL,0x3f930d388dab5e13ULL,0x3f91490334603012ULL,0x3f8f152a4f72dd49ULL,0x3f8ba48d274f8facULL,0x3f8841040d8da478ULL,0x3f84eb96421acfe0ULL,0x3f81a59229952f92ULL,0x3f7ce160f8ec6837ULL,0x3f769ea8d90cb85dULL,0x3f708a1f03b0b1fdULL,0x3f655f9f43c1b067ULL,0x3f54a605b6b9f70fULL};
static const uint64_t zig_nor_r_bits = 0x400d3bb48209ad33ULL;
static const uint64_t zig_neg_inv_bits = 0xbfd183aa6c20e8c1ULL;

typedef __uint128_t u128;
typedef struct { u128 state, inc; } pcg64_t;
#define PCG_MUL ((((u128)2549297995355413924ULL) << 64) | 4865540595714422341ULL)
static inline void pcg_step(pcg64_t* r){ r->state = r->state * PCG_MUL + r->inc; }
static inline uint64_t rotr64(uint64_t v, unsigned rot){ return (v >> rot) | (v << ((64u - rot) & 63u)); }
static inline uint64_t pcg_next64(pcg64_t* r){ pcg_step(r); u128 s = r->state; return rotr64((uint64_t)(s >> 64) ^ (uint64_t)s, (unsigned)(s >> 122)); }
static inline double pcg_nextd(pcg64_t* r){ return (double)(pcg_next64(r) >> 11) * (1.0/9007199254740992.0); }
static void pcg_seed(pcg64_t* r, const uint64_t* w){
  u128 initstate = (((u128)w[0]) << 64) | w[1];
  u128 initseq   = (((u128)w[2]) << 64) | w[3];
  r->state = 0; r->inc = (initseq << 1) | 1;
  pcg_step(r);
  r->state += initstate;
  pcg_step(r);
}
static const double* wi_double = (const double*)wi_double_bits;
static const double* fi_double = (const double*)fi_double_bits;
static double wi_signed[512];  // [sign<<8 | idx]; filled in setup()
static void init_wi_signed(void){
  for (int i = 0; i < 256; i++){
    uint64_t b = wi_double_bits[i];
    wi_signed[i] = *(double*)&b;
    uint64_t bn = b ^ 0x8000000000000000ULL;
    wi_signed[256 + i] = *(double*)&bn;
  }
}
static inline double znorm(pcg64_t* rng){
  for (;;){
    uint64_t r = pcg_next64(rng);
    int idx = (int)(r & 0xff);
    uint64_t r9 = r >> 9;
    uint64_t rabs = r9 & 0x000fffffffffffffULL;
    double x = (double)rabs * wi_double[idx];
    if (r & 0x100) x = -x;
    if (__builtin_expect(rabs < ki_double[idx], 1)) return x;
    if (idx == 0){
      const double nor_r = *(const double*)&zig_nor_r_bits;
      const double neg_inv = *(const double*)&zig_neg_inv_bits;
      for (;;){
        double xx = neg_inv * log1p(-pcg_nextd(rng));
        double yy = -log1p(-pcg_nextd(rng));
        if (yy + yy > xx * xx){
          double v = xx + nor_r;
          return (r9 & 0x100) ? -v : v;
        }
      }
    } else {
      if ((fi_double[idx-1] - fi_double[idx]) * pcg_nextd(rng) + fi_double[idx] < exp(-0.5*x*x))
        return x;
    }
  }
}
void fill_normals(const uint64_t* w, long n, double* out){
  pcg64_t r; pcg_seed(&r, w);
  for (long i = 0; i < n; i++) out[i] = znorm(&r);
}
// ---- fast path: 4-way jump-ahead raw generation + buffered ziggurat ----
typedef struct {
  u128 s[8], a8, c8;
  uint64_t* p; uint64_t* end;
  uint64_t buf[16384];
} pcg4_t;
static void pcg4_init(pcg4_t* g, const uint64_t* w){
  pcg64_t r; pcg_seed(&r, w);
  u128 M = PCG_MUL, inc = r.inc, s = r.state;
  for (int j = 0; j < 8; j++){ s = s * M + inc; g->s[j] = s; }
  u128 a2 = M * M, c2 = (M + 1) * inc;
  u128 a4 = a2 * a2, c4 = (a2 + 1) * c2;
  g->a8 = a4 * a4;
  g->c8 = (a4 + 1) * c4;
  g->p = g->end = g->buf;
}
static inline uint64_t pcg_out_s(u128 s){
  return rotr64((uint64_t)(s >> 64) ^ (uint64_t)s, (unsigned)(s >> 122));
}
static void __attribute__((noinline)) pcg4_refill(pcg4_t* g){
  u128 a = g->a8, c = g->c8;
  uint64_t* b = g->buf;
  u128 s0 = g->s[0], s1 = g->s[1], s2 = g->s[2], s3 = g->s[3];
  u128 s4 = g->s[4], s5 = g->s[5], s6 = g->s[6], s7 = g->s[7];
  for (long t = 0; t < 16384; t += 8){
    b[t+0] = pcg_out_s(s0); b[t+1] = pcg_out_s(s1); b[t+2] = pcg_out_s(s2); b[t+3] = pcg_out_s(s3);
    b[t+4] = pcg_out_s(s4); b[t+5] = pcg_out_s(s5); b[t+6] = pcg_out_s(s6); b[t+7] = pcg_out_s(s7);
    s0 = s0*a + c; s1 = s1*a + c; s2 = s2*a + c; s3 = s3*a + c;
    s4 = s4*a + c; s5 = s5*a + c; s6 = s6*a + c; s7 = s7*a + c;
  }
  g->s[0] = s0; g->s[1] = s1; g->s[2] = s2; g->s[3] = s3;
  g->s[4] = s4; g->s[5] = s5; g->s[6] = s6; g->s[7] = s7;
  g->p = b; g->end = b + 16384;
}
static inline uint64_t nraw(pcg4_t* g){
  if (__builtin_expect(g->p == g->end, 0)) pcg4_refill(g);
  return *g->p++;
}
static inline double nrawd(pcg4_t* g){ return (double)(nraw(g) >> 11) * (1.0/9007199254740992.0); }
static inline double znorm4(pcg4_t* g){
  for (;;){
    uint64_t r = nraw(g);
    int idx = (int)(r & 0xff);
    uint64_t r9 = r >> 9;
    uint64_t rabs = r9 & 0x000fffffffffffffULL;
    double x = (double)rabs * wi_signed[r & 0x1ff];
    if (__builtin_expect(rabs < ki_double[idx], 1)) return x;
    if (idx == 0){
      const double nor_r = *(const double*)&zig_nor_r_bits;
      const double neg_inv = *(const double*)&zig_neg_inv_bits;
      for (;;){
        double xx = neg_inv * log1p(-nrawd(g));
        double yy = -log1p(-nrawd(g));
        if (yy + yy > xx * xx){
          double v = xx + nor_r;
          return (r9 & 0x100) ? -v : v;
        }
      }
    } else {
      if ((fi_double[idx-1] - fi_double[idx]) * nrawd(g) + fi_double[idx] < exp(-0.5*x*x))
        return x;
    }
  }
}
static pcg4_t G4A, G4B;
void fill_normals_fast(const uint64_t* w, long n, double* out){
  pcg4_t* g = &G4A;
  pcg4_init(g, w);
  long i = 0;
  while (i + 4 <= n){
    if (__builtin_expect(g->p + 4 > g->end, 0)) { out[i++] = znorm4(g); continue; }
    const uint64_t* p = g->p;
    uint64_t r0 = p[0], r1 = p[1], r2 = p[2], r3 = p[3];
    uint64_t a0 = (r0 >> 9) & 0x000fffffffffffffULL;
    uint64_t a1 = (r1 >> 9) & 0x000fffffffffffffULL;
    uint64_t a2 = (r2 >> 9) & 0x000fffffffffffffULL;
    uint64_t a3 = (r3 >> 9) & 0x000fffffffffffffULL;
    int i0 = (int)(r0 & 0xff), i1 = (int)(r1 & 0xff), i2 = (int)(r2 & 0xff), i3 = (int)(r3 & 0xff);
    int ok = (a0 < ki_double[i0]) & (a1 < ki_double[i1]) & (a2 < ki_double[i2]) & (a3 < ki_double[i3]);
    if (__builtin_expect(ok, 1)){
      out[i]   = (double)a0 * wi_signed[r0 & 0x1ff];
      out[i+1] = (double)a1 * wi_signed[r1 & 0x1ff];
      out[i+2] = (double)a2 * wi_signed[r2 & 0x1ff];
      out[i+3] = (double)a3 * wi_signed[r3 & 0x1ff];
      g->p += 4; i += 4;
    } else {
      out[i] = znorm4(g);
      i++;
    }
  }
  for (; i < n; i++) out[i] = znorm4(g);
}
static void fill_normals2(const uint64_t* w1, long n1, double* o1, const uint64_t* w2, long n2, double* o2){
  fill_normals_fast(w1, n1, o1);
  fill_normals_fast(w2, n2, o2);
}
static double *Gxr, *Gcr; static size_t Gcap = 0;
static void gens_ensure(size_t n){
  if (n <= Gcap) return;
  free(Gxr); free(Gcr);
  Gxr = malloc(2*n*sizeof(double));
  Gcr = malloc(2*n*sizeof(double));
  Gcap = n;
}

static inline __m512d map8_pre(__m512d R, __m512d I){
  __m512d r2 = _mm512_fmadd_pd(I, I, _mm512_mul_pd(R, R));
  __m512d y = _mm512_rsqrt14_pd(r2);
  __m512d h = _mm512_mul_pd(r2, _mm512_set1_pd(0.5));
  __m512d t = _mm512_mul_pd(h, y);
  y = _mm512_mul_pd(y, _mm512_fnmadd_pd(t, y, _mm512_set1_pd(1.5)));
  t = _mm512_mul_pd(h, y);
  y = _mm512_mul_pd(y, _mm512_fnmadd_pd(t, y, _mm512_set1_pd(1.5)));
  __m512d r = _mm512_mul_pd(r2, y);
  return _mm512_add_pd(r, _mm512_set1_pd(1.0));
}
static inline void map8(V8 zr, V8 zi, V8* mr, V8* mi){   // hw divide (divider unit)
  __m512d R = (__m512d)zr, I = (__m512d)zi;
  __m512d dd = map8_pre(R, I);
  __m512d w = _mm512_div_pd(_mm512_set1_pd(1.0), dd);
  *mr = (V8)_mm512_mul_pd(R, w);
  *mi = (V8)_mm512_mul_pd(I, w);
}
static inline void map8nr(V8 zr, V8 zi, V8* mr, V8* mi){ // Newton reciprocal (FMA ports)
  __m512d R = (__m512d)zr, I = (__m512d)zi;
  __m512d dd = map8_pre(R, I);
  __m512d w = _mm512_rcp14_pd(dd);
  w = _mm512_mul_pd(w, _mm512_fnmadd_pd(dd, w, _mm512_set1_pd(2.0)));
  w = _mm512_mul_pd(w, _mm512_fnmadd_pd(dd, w, _mm512_set1_pd(2.0)));
  *mr = (V8)_mm512_mul_pd(R, w);
  *mi = (V8)_mm512_mul_pd(I, w);
}
#define MAPCALL map8

// interleave 8 re + 8 im -> 16 doubles (8 complex)
static inline void ilv8(__m512d re, __m512d im, double* out){
  __m512d lo = _mm512_unpacklo_pd(re, im);   // r0 i0 r2 i2 r4 i4 r6 i6
  __m512d hi = _mm512_unpackhi_pd(re, im);   // r1 i1 r3 i3 r5 i5 r7 i7
  const __m512i AL = _mm512_setr_epi64(0,1,8,9,2,3,10,11);
  const __m512i AH = _mm512_setr_epi64(4,5,12,13,6,7,14,15);
  _mm512_storeu_pd(out,     _mm512_permutex2var_pd(lo, AL, hi));
  _mm512_storeu_pd(out + 8, _mm512_permutex2var_pd(lo, AH, hi));
}
static inline void ilv8_tail(__m512d re, __m512d im, double* out, int n){ // n complex < 8
  __m512d lo = _mm512_unpacklo_pd(re, im);
  __m512d hi = _mm512_unpackhi_pd(re, im);
  const __m512i AL = _mm512_setr_epi64(0,1,8,9,2,3,10,11);
  const __m512i AH = _mm512_setr_epi64(4,5,12,13,6,7,14,15);
  unsigned mask = (1u << (2*n)) - 1u;
  _mm512_mask_storeu_pd(out, (__mmask8)mask, _mm512_permutex2var_pd(lo, AL, hi));
  if (n > 4) _mm512_mask_storeu_pd(out + 8, (__mmask8)(mask >> 8), _mm512_permutex2var_pd(lo, AH, hi));
}
// 8x8 double transpose: in = 8 rows at p, p+rs, ...; out[c] = column c
static inline void tr8(const double* p, ptrdiff_t rs, V8* o){
  __m512d r0 = _mm512_loadu_pd(p + 0*rs);
  __m512d r1 = _mm512_loadu_pd(p + 1*rs);
  __m512d r2 = _mm512_loadu_pd(p + 2*rs);
  __m512d r3 = _mm512_loadu_pd(p + 3*rs);
  __m512d r4 = _mm512_loadu_pd(p + 4*rs);
  __m512d r5 = _mm512_loadu_pd(p + 5*rs);
  __m512d r6 = _mm512_loadu_pd(p + 6*rs);
  __m512d r7 = _mm512_loadu_pd(p + 7*rs);
  __m512d s0 = _mm512_unpacklo_pd(r0, r1), s1 = _mm512_unpackhi_pd(r0, r1);
  __m512d s2 = _mm512_unpacklo_pd(r2, r3), s3 = _mm512_unpackhi_pd(r2, r3);
  __m512d s4 = _mm512_unpacklo_pd(r4, r5), s5 = _mm512_unpackhi_pd(r4, r5);
  __m512d s6 = _mm512_unpacklo_pd(r6, r7), s7 = _mm512_unpackhi_pd(r6, r7);
  const __m512i I2L = _mm512_setr_epi64(0,1,8,9,4,5,12,13);
  const __m512i I2H = _mm512_setr_epi64(2,3,10,11,6,7,14,15);
  __m512d t0 = _mm512_permutex2var_pd(s0, I2L, s2);
  __m512d t1 = _mm512_permutex2var_pd(s0, I2H, s2);
  __m512d t2 = _mm512_permutex2var_pd(s1, I2L, s3);
  __m512d t3 = _mm512_permutex2var_pd(s1, I2H, s3);
  __m512d t4 = _mm512_permutex2var_pd(s4, I2L, s6);
  __m512d t5 = _mm512_permutex2var_pd(s4, I2H, s6);
  __m512d t6 = _mm512_permutex2var_pd(s5, I2L, s7);
  __m512d t7 = _mm512_permutex2var_pd(s5, I2H, s7);
  const __m512i I4L = _mm512_setr_epi64(0,1,2,3,8,9,10,11);
  const __m512i I4H = _mm512_setr_epi64(4,5,6,7,12,13,14,15);
  o[0] = (V8)_mm512_permutex2var_pd(t0, I4L, t4);
  o[4] = (V8)_mm512_permutex2var_pd(t0, I4H, t4);
  o[2] = (V8)_mm512_permutex2var_pd(t1, I4L, t5);
  o[6] = (V8)_mm512_permutex2var_pd(t1, I4H, t5);
  o[1] = (V8)_mm512_permutex2var_pd(t2, I4L, t6);
  o[5] = (V8)_mm512_permutex2var_pd(t2, I4H, t6);
  o[3] = (V8)_mm512_permutex2var_pd(t3, I4L, t7);
  o[7] = (V8)_mm512_permutex2var_pd(t3, I4H, t7);
}
'''

def emit_map_codelet(n, width=8):
    """codelet with fused (+c, map) at stores"""
    import gen as G
    d = G.Dag()
    xs = [(d.inp(('r', j)), d.inp(('i', j))) for j in range(n)]
    X = G.build_dft(d, xs, -1)
    ty, K = G.WIDTHS[width]
    name = f"fftmap{n}_w{width}"
    lines = []
    lines.append(f"static void {name}(const double* ri, const double* ii, ptrdiff_t is, double* ro, double* io, ptrdiff_t os, const double* cr, const double* ci) {{")
    var = {}
    for j in range(n):
        ir, im_ = xs[j]
        var[ir] = f"xr{j}"; var[im_] = f"xi{j}"
        lines.append(f"  {ty} xr{j} = LD{width}(ri + {j}*is); {ty} xi{j} = LD{width}(ii + {j}*is);")
    roots = [i for xy in X for i in xy]
    order = []
    state = {}
    def visit(i):
        st = state.get(i, 0)
        if st: return
        state[i] = 1
        t = d.nodes[i]
        for ch in t[1:]:
            if isinstance(ch, int): visit(ch)
        order.append(i)
        state[i] = 2
    for r in roots: visit(r)
    for i in order:
        if i in var: continue
        t = d.nodes[i]
        k = t[0]
        if k == 'in': continue
        if k == 'zero':
            var[i] = f"{K}(0.0)"; continue
        v = f"t{i}"
        if k == 'add': lines.append(f"  {ty} {v} = {var[t[1]]} + {var[t[2]]};")
        elif k == 'sub': lines.append(f"  {ty} {v} = {var[t[1]]} - {var[t[2]]};")
        elif k == 'neg': lines.append(f"  {ty} {v} = -{var[t[1]]};")
        elif k == 'mul': lines.append(f"  {ty} {v} = {K}({float(t[1][1])!r}) * {var[t[2]]};")
        var[i] = v
    for kk in range(n):
        r, im_ = X[kk]
        lines.append(f"  {{ V8 zr = {var[r]} + LD8(cr + {kk}*os); V8 zi = {var[im_]} + LD8(ci + {kk}*os); V8 mr, mi; MAPCALL(zr, zi, &mr, &mi); ST8(ro + {kk}*os, mr); ST8(io + {kk}*os, mi); }}")
    lines.append("}")
    return "\n".join(lines)



def drv_for(L):
    XS = XSMAP[L]
    ZP = ZPMAP[L]
    NROWS = L * L
    big = L in BIG
    if big:
        yc = [(s, 8) for s in range(0, ((L + 7)//8)*8, 8)]
    else:
        yc = ychunks(L)
    nblk = (L + 7) // 8
    s = []
    A = lambda t: s.append(t)
    # ---- A_y: src -> dst (out-of-place; overlap chunks) ----
    A(f"static void ay_{L}(const double* sr, const double* si, double* dr, double* di){{")
    A(f"  for (int x = 0; x < {L}; x++) {{")
    A(f"    const double* pr = sr + (size_t)x*{XS}; const double* pi = si + (size_t)x*{XS};")
    A(f"    double* qr = dr + (size_t)x*{XS}; double* qi = di + (size_t)x*{XS};")
    for (st, w) in yc:
        fn = f"fft{L}_yy" if w == 8 else f"fft{L}_yy4"
        A(f"    {fn}(pr + {st}, pi + {st}, qr + {st}, qi + {st});")
    A("  }")
    A("}")
    if big:
        A(f"static void ayp_{L}(const double* pr, const double* pi, double* qr, double* qi){{")
        for (st, w) in yc:
            A(f"  fft{L}_yy(pr + {st}, pi + {st}, qr + {st}, qi + {st});")
        A("}")
    # ---- A_x ----
    if big:
        if L in (36, 64):
            r = {36: 6, 64: 8}[L]
            # odd steps: input sigma=id: stage1 strided, stage2 consec
            if L == 64:
                ploop = f"for (int y = 0; y < 64; y++) for (int zc = 0; zc < 64; zc += 8) {{ int p = y*{ZP} + zc;"
            else:
                ploop = f"for (int p = 0; p < {L*ZP}; p += 8) {{"
            A(f"static void axmap_{L}_odd(double* sr, double* si, const double* cr, const double* ci){{")
            for g in range(r):
                A(f"  {ploop} ax1_{L}_strided_{g}(sr + (size_t){g}*{XS} + p, si + (size_t){g}*{XS} + p); }}")
            A(f"  for (int k1 = 0; k1 < {r}; k1++)")
            A(f"    {ploop} size_t o = (size_t)k1*{r*XS} + p; ax2_{L}_consec(sr + o, si + o, cr + o, ci + o); }}")
            A(f"}}")
            # even steps: input sigma=rho: stage1 consec, stage2 strided
            A(f"static void axmap_{L}_even(double* sr, double* si, const double* cr, const double* ci){{")
            for g in range(r):
                A(f"  {ploop} ax1_{L}_consec_{g}(sr + (size_t){g}*{r*XS} + p, si + (size_t){g}*{r*XS} + p); }}")
            A(f"  for (int k1 = 0; k1 < {r}; k1++)")
            A(f"    {ploop} size_t o = (size_t)k1*{XS} + p; ax2_{L}_strided(sr + o, si + o, cr + o, ci + o); }}")
            A(f"}}")
        elif L == 45:
            # fused alternating-factorization path: helpers
            # odd steps (sigma=id, (N1,N2)=(5,9)): stage1 groups g<9 read {9a+g} (stride 9); stage2 groups k1<5 consecutive [9k1,9k1+9)
            # even steps (sigma=rhoA, (N1,N2)=(9,5)): stage1 groups b<5 consecutive [9b,9b+9); stage2 groups k1<9 strided {k1+9b}
            A(f"static void ayp_45(const double* pr, const double* pi, double* qr, double* qi);")
            A(f"static void azp_45(const double* wr, const double* wi, double* tr, double* ti);")
            A(f"static void grp_planes1_45(double* sr, double* si, double* wr, double* wi, int g, int odd){{")
            A(f"  int cnt = odd ? 5 : 9;")
            A(f"  for (int a = 0; a < cnt; a++) {{")
            A(f"    size_t o = odd ? (size_t)(9*a + g)*{XS} : (size_t)(9*g + a)*{XS};")
            A(f"    ayp_45(sr + o, si + o, wr, wi);")
            A(f"    azp_45(wr, wi, sr + o, si + o);")
            A(f"  }}")
            A(f"  if (odd) switch (g) {{")
            for g in range(9):
                A(f"    case {g}: for (int p = 0; p < {L*ZP}; p += 8) ax1_45_s19_{g}(sr + (size_t){g}*{XS} + p, si + (size_t){g}*{XS} + p); break;")
            A(f"  }} else switch (g) {{")
            for g in range(5):
                A(f"    case {g}: for (int p = 0; p < {L*ZP}; p += 8) ax1_45_s95_{g}(sr + (size_t){g}*{9*XS} + p, si + (size_t){g}*{9*XS} + p); break;")
            A(f"  }}")
            A(f"}}")
            A(f"static void grp_stage2_45(double* sr, double* si, const double* cr, const double* ci, int k1, int odd){{")
            A(f"  if (odd) {{ for (int p = 0; p < {L*ZP}; p += 8) {{ size_t o = (size_t)k1*{9*XS} + p; ax2_45_s19(sr + o, si + o, cr + o, ci + o); }} }}")
            A(f"  else {{ for (int p = 0; p < {L*ZP}; p += 8) {{ size_t o = (size_t)k1*{XS} + p; ax2_45_s95(sr + o, si + o, cr + o, ci + o); }} }}")
            A(f"}}")
        else:
            A(f"static void axmap_{L}(double* sr, double* si, const double* cr, const double* ci){{")
            A(f"  for (int p = 0; p < {L*ZP}; p += 8)")
            A(f"    fftmap{L}_pp(sr + p, si + p, sr + p, si + p, cr + p, ci + p);")
            A("}")
    else:
        A(f"static void ax_{L}(const double* sr, const double* si, double* dr, double* di){{")
        A(f"  for (int p = 0; p + 8 <= {L*ZP}; p += 8)")
        A(f"    fft{L}_pp(sr + p, si + p, dr + p, di + p);")
        if (L*ZP) % 8:
            t = L*ZP - 8
            A(f"  fft{L}_pp(sr + {t}, si + {t}, dr + {t}, di + {t});")
        A("}")
    # ---- A_z ----
    if big:
        # per-plane: W((y,z) stride L) -> plane ((z,y) stride ZP), plain strided stores
        A(f"static void azp_{L}(const double* wr, const double* wi, double* tr, double* ti){{")
        A(f"  V8 BR[{nblk*8}] __attribute__((aligned(64))); V8 BI[{nblk*8}] __attribute__((aligned(64)));")
        starts = list(range(0, ((L + 7)//8)*8, 8))
        A(f"  static const int y0s[{len(starts)}] = {{{', '.join(map(str, starts))}}};")
        A(f"  for (int bi = 0; bi < {len(starts)}; bi++) {{")
        A(f"    int y0 = y0s[bi];")
        A(f"    const double* p = wr + (size_t)y0*{ZP};")
        A(f"    const double* q = wi + (size_t)y0*{ZP};")
        for zb in range(0, L, 8):
            A(f"    tr8(p + {zb}, {ZP}, BR + {zb}); tr8(q + {zb}, {ZP}, BI + {zb});")
        A(f"    fft{L}_zz((const double*)BR, (const double*)BI, tr + y0, ti + y0);")
        A("  }")
        A("}")
    else:
        # rotation: bundle rows r0: store lanes contiguous at T + k*XS + r0, c fused
        A(f"static void az_{L}(const double* sr, const double* si, double* tr, double* ti, const double* cr, const double* ci){{")
        A(f"  V8 BR[{nblk*8}] __attribute__((aligned(64))); V8 BI[{nblk*8}] __attribute__((aligned(64)));")
        A(f"  for (int r0 = 0; ; ) {{")
        A(f"    const double* p = sr + (size_t)r0*{L};")
        A(f"    const double* q = si + (size_t)r0*{L};")
        for zb in range(0, L, 8):
            A(f"    tr8(p + {zb}, {L}, BR + {zb}); tr8(q + {zb}, {L}, BI + {zb});")
        A(f"    fftmap{L}_zz((const double*)BR, (const double*)BI, tr + r0, ti + r0, cr + r0, ci + r0);")
        A(f"    if (r0 + 8 >= {NROWS}) break;")
        A(f"    r0 += 8; if (r0 + 8 > {NROWS}) r0 = {NROWS} - 8;")
        A(f"  }}")
        A("}")
    # ---- step ----
    if big:
        if L in (36, 64):
            r = {36: 6, 64: 8}[L]
            if L == 64:
                ploop = f"for (int y = 0; y < 64; y++) for (int zc = 0; zc < 64; zc += 8) {{ int p = y*{ZP} + zc;"
            else:
                ploop = f"for (int p = 0; p < {L*ZP}; p += 8) {{"
            A(f"static void step_{L}(double* sr, double* si, double* wr, double* wi, const double* ccr, const double* cci, int odd){{")
            A(f"  if (odd) {{")
            A(f"    for (int g = 0; g < {r}; g++) {{")
            A(f"      for (int a = 0; a < {r}; a++) {{")
            A(f"        size_t o = (size_t)({r}*a + g)*{XS};")
            A(f"        ayp_{L}(sr + o, si + o, wr, wi);")
            A(f"        azp_{L}(wr, wi, sr + o, si + o);")
            A(f"      }}")
            A(f"      switch (g) {{")
            for g in range(r):
                A(f"        case {g}: {ploop} ax1_{L}_strided_{g}(sr + (size_t){g}*{XS} + p, si + (size_t){g}*{XS} + p); }} break;")
            A(f"      }}")
            A(f"    }}")
            A(f"    for (int k1 = 0; k1 < {r}; k1++)")
            A(f"      {ploop} size_t o = (size_t)k1*{r*XS} + p; ax2_{L}_consec(sr + o, si + o, ccr + o, cci + o); }}")
            A(f"  }} else {{")
            A(f"    for (int g = 0; g < {r}; g++) {{")
            A(f"      for (int a = 0; a < {r}; a++) {{")
            A(f"        size_t o = (size_t)({r}*g + a)*{XS};")
            A(f"        ayp_{L}(sr + o, si + o, wr, wi);")
            A(f"        azp_{L}(wr, wi, sr + o, si + o);")
            A(f"      }}")
            A(f"      switch (g) {{")
            for g in range(r):
                A(f"        case {g}: {ploop} ax1_{L}_consec_{g}(sr + (size_t){g}*{r*XS} + p, si + (size_t){g}*{r*XS} + p); }} break;")
            A(f"      }}")
            A(f"    }}")
            A(f"    for (int k1 = 0; k1 < {r}; k1++)")
            A(f"      {ploop} size_t o = (size_t)k1*{XS} + p; ax2_{L}_strided(sr + o, si + o, ccr + o, cci + o); }}")
            A(f"  }}")
            A(f"}}")

        elif L != 45:
            A(f"static void step_{L}(double* sr, double* si, double* wr, double* wi, const double* ccr, const double* cci){{")
            A(f"  for (int x = 0; x < {L}; x++) {{")
            A(f"    size_t o = (size_t)x*{XS};")
            A(f"    ayp_{L}(sr + o, si + o, wr, wi);")
            A(f"    azp_{L}(wr, wi, sr + o, si + o);")
            A(f"  }}")
            A(f"  axmap_{L}(sr, si, ccr, cci);")
            A(f"}}")
    else:
        A(f"static void step_{L}(double* cr_, double* ci_, double* tr, double* ti, const double* ccr, const double* cci){{")
        A(f"  ay_{L}(cr_, ci_, tr, ti);")
        A(f"  ax_{L}(tr, ti, cr_, ci_);")
        A(f"  az_{L}(cr_, ci_, tr, ti, ccr, cci);")
        A(f"}}")
    return "\n".join(s)

def io_for(L):
    XS = XSMAP[L]; ZP = ZPMAP[L]
    big = L in BIG
    s = []
    A = lambda t: s.append(t)
    A(f"static void load_{L}(const double* xr, const double* xi, double scale, double* rr, double* ri){{")
    A(f"  for (int x = 0; x < {L}; x++) for (int y = 0; y < {L}; y++) {{")
    A(f"    const double* pr = xr + (size_t)(((x*{L})+y)*{L});")
    A(f"    const double* pi = xi + (size_t)(((x*{L})+y)*{L});")
    A(f"    double* ar = rr + (size_t)x*{XS} + (size_t)y*{ZP};")
    A(f"    double* ai = ri + (size_t)x*{XS} + (size_t)y*{ZP};")
    A(f"    for (int z = 0; z < {L}; z++) {{ ar[z] = scale*pr[z]; ai[z] = scale*pi[z]; }}")
    A(f"  }}")
    A(f"}}")
    if big:
        # cB layout: (x,z,y) swap
        A(f"static void loadB_{L}(const double* xr, const double* xi, double scale, double* rr, double* ri){{")
        A(f"  V8 BUF[8] __attribute__((aligned(64)));")
        A(f"  __m512d SC = _mm512_set1_pd(scale);")
        A(f"  for (int x = 0; x < {L}; x++) {{")
        if L in (36, 64):
            r0 = {36: 6, 64: 8}[L]
            A(f"    int xq = {r0}*(x % {r0}) + x / {r0};")
        elif L == 45:
            A(f"    int xq = (x == 44) ? 44 : (9*x) % 44;")
        else:
            A(f"    int xq = x;")
        A(f"    for (int h = 0; h < 2; h++) {{")
        A(f"      const double* src = h ? xi : xr;")
        A(f"      double* dst = h ? ri : rr;")
        A(f"      for (int y0 = 0; y0 + 8 <= {L}; y0 += 8) {{")
        A(f"        for (int zb = 0; zb < {L}; zb += 8) {{")
        A(f"          int zn = {L} - zb; if (zn > 8) zn = 8;")
        A(f"          tr8(src + (size_t)(((x*{L})+y0)*{L}) + zb, {L}, BUF);")
        A(f"          for (int t = 0; t < zn; t++) {{")
        A(f"            __m512d v = _mm512_mul_pd(SC, (__m512d)BUF[t]);")
        A(f"            _mm512_storeu_pd(dst + (size_t)xq*{XS} + (size_t)(zb+t)*{ZP} + y0, v);")
        A(f"          }}")
        A(f"        }}")
        A(f"      }}")
        A(f"      for (int y = {L} - ({L} % 8); y < {L}; y++) {{")
        A(f"        const double* p = src + (size_t)(((x*{L})+y)*{L});")
        A(f"        double* a = dst + (size_t)xq*{XS} + y;")
        A(f"        for (int z = 0; z < {L}; z++) a[(size_t)z*{ZP}] = scale*p[z];")
        A(f"      }}")
        A(f"    }}")
        A(f"  }}")
        A(f"}}")
    else:
        # cB layout: (z,x,y): cB[(z*L+x)*L+y] = c[x][y][z]
        A(f"static void loadB_{L}(const double* xr, const double* xi, double scale, double* rr, double* ri){{")
        A(f"  for (int x = 0; x < {L}; x++) for (int y = 0; y < {L}; y++) {{")
        A(f"    const double* pr = xr + (size_t)(((x*{L})+y)*{L});")
        A(f"    const double* pi = xi + (size_t)(((x*{L})+y)*{L});")
        A(f"    size_t base = (size_t)x*{L} + y;")
        A(f"    for (int z = 0; z < {L}; z++) {{ rr[base + (size_t)z*{XS}] = scale*pr[z]; ri[base + (size_t)z*{XS}] = scale*pi[z]; }}")
        A(f"  }}")
        A(f"}}")
        # cC layout: (y,z,x): cC[(y*L+z)*L+x] = c[x][y][z]
        A(f"static void loadC_{L}(const double* xr, const double* xi, double scale, double* rr, double* ri){{")
        A(f"  for (int x = 0; x < {L}; x++) for (int y = 0; y < {L}; y++) {{")
        A(f"    const double* pr = xr + (size_t)(((x*{L})+y)*{L});")
        A(f"    const double* pi = xi + (size_t)(((x*{L})+y)*{L});")
        A(f"    double* ar = rr + (size_t)y*{XS} + x;")
        A(f"    double* ai = ri + (size_t)y*{XS} + x;")
        A(f"    for (int z = 0; z < {L}; z++) {{ ar[(size_t)z*{L}] = scale*pr[z]; ai[(size_t)z*{L}] = scale*pi[z]; }}")
        A(f"  }}")
        A(f"}}")
    if big and L in (36, 45, 64):
        A(f"static void emitpl1_{L}(const double* ar, const double* ai, double* p);")
        A(f"static void emitp1_{L}(const double* rr, const double* ri, int q, double* out){{")
        if L in (36, 64):
            r0 = {36: 6, 64: 8}[L]
            A(f"  int x = {r0}*(q % {r0}) + q / {r0};")
        else:
            A(f"  int x = (q == 44) ? 44 : (5*q) % 44;")
        A(f"  emitpl1_{L}(rr + (size_t)q*{XS}, ri + (size_t)q*{XS}, out + 2*(size_t)x*{L*L});")
        A(f"}}")
    # emit: orient 0: (x,y,z); big orient 1: (x,z,y); small orient 1: (z,x,y), orient 2: (y,z,x)
    if big:
        # vectorized per-plane helpers
        A(f"static void emitpl0_{L}(const double* ar, const double* ai, double* p){{")
        A(f"  // (y,z) plane, rows stride {ZP} -> interleaved out rows of 2*{L}")
        A(f"  for (int y = 0; y < {L}; y++) {{")
        A(f"    const double* r = ar + (size_t)y*{ZP}; const double* q = ai + (size_t)y*{ZP};")
        A(f"    double* o = p + 2*(size_t)y*{L};")
        nfull = L // 8
        for zb in range(0, (L//8)*8, 8):
            A(f"    ilv8(_mm512_loadu_pd(r + {zb}), _mm512_loadu_pd(q + {zb}), o + {2*zb});")
        if L % 8:
            zb = (L//8)*8
            A(f"    ilv8_tail(_mm512_loadu_pd(r + {zb}), _mm512_loadu_pd(q + {zb}), o + {2*zb}, {L%8});")
        A(f"  }}")
        A(f"}}")
        A(f"static void emitpl1_{L}(const double* ar, const double* ai, double* p){{")
        A(f"  // (z,y) plane (rows stride {ZP} over z) -> out[y][z] interleaved; transpose via tr8")
        A(f"  V8 TR[8] __attribute__((aligned(64))); V8 TI[8] __attribute__((aligned(64)));")
        A(f"  for (int yb = 0; yb < {L}; yb += 8) {{")
        A(f"    int yn = {L} - yb; if (yn > 8) yn = 8;")
        A(f"    for (int zb = 0; zb < {L}; zb += 8) {{")
        A(f"      int zn = {L} - zb; if (zn > 8) zn = 8;")
        A(f"      tr8(ar + (size_t)zb*{ZP} + yb, {ZP}, TR);")
        A(f"      tr8(ai + (size_t)zb*{ZP} + yb, {ZP}, TI);")
        A(f"      for (int t = 0; t < yn; t++) {{")
        A(f"        double* o = p + 2*((size_t)(yb+t)*{L} + zb);")
        A(f"        if (zn == 8) ilv8((__m512d)TR[t], (__m512d)TI[t], o); else ilv8_tail((__m512d)TR[t], (__m512d)TI[t], o, zn);")
        A(f"      }}")
        A(f"    }}")
        A(f"  }}")
        A(f"}}")
        A(f"static void emit_{L}(const double* rr, const double* ri, int orient, double* out){{")
        A(f"  if (orient == 0) {{")
        A(f"    for (int x = 0; x < {L}; x++)")
        A(f"      emitpl0_{L}(rr + (size_t)x*{XS}, ri + (size_t)x*{XS}, out + 2*(size_t)x*{L*L});")
        A(f"  }} else {{")
        A(f"    for (int x = 0; x < {L}; x++) {{")
        if L in (36, 64):
            r0 = {36: 6, 64: 8}[L]
            A(f"      int xq = {r0}*(x % {r0}) + x / {r0};")
        elif L == 45:
            A(f"      int xq = (x == 44) ? 44 : (9*x) % 44;")
        else:
            A(f"      int xq = x;")
        A(f"      emitpl1_{L}(rr + (size_t)xq*{XS}, ri + (size_t)xq*{XS}, out + 2*(size_t)x*{L*L});")
        A(f"    }}")
        A(f"  }}")
        A(f"}}")
    else:
        A(f"static void emit_{L}(const double* rr, const double* ri, int orient, double* out){{")
        A(f"  if (orient == 0) {{")
        A(f"    for (int x = 0; x < {L}; x++) for (int y = 0; y < {L}; y++) {{")
        A(f"      const double* ar = rr + (size_t)x*{XS} + (size_t)y*{ZP};")
        A(f"      const double* ai = ri + (size_t)x*{XS} + (size_t)y*{ZP};")
        A(f"      double* p = out + 2*(size_t)(((x*{L})+y)*{L});")
        A(f"      for (int z = 0; z < {L}; z++) {{ p[2*z] = ar[z]; p[2*z+1] = ai[z]; }}")
        A(f"    }} }}")
        A(f"  else if (orient == 1) {{")
        A(f"    for (int x = 0; x < {L}; x++) for (int y = 0; y < {L}; y++) {{")
        A(f"      size_t base = (size_t)x*{L} + y;")
        A(f"      double* p = out + 2*(size_t)(((x*{L})+y)*{L});")
        A(f"      for (int z = 0; z < {L}; z++) {{ p[2*z] = rr[base + (size_t)z*{XS}]; p[2*z+1] = ri[base + (size_t)z*{XS}]; }}")
        A(f"    }} }}")
        A(f"  else {{")
        A(f"    for (int x = 0; x < {L}; x++) for (int y = 0; y < {L}; y++) {{")
        A(f"      const double* ar = rr + (size_t)y*{XS} + x;")
        A(f"      const double* ai = ri + (size_t)y*{XS} + x;")
        A(f"      double* p = out + 2*(size_t)(((x*{L})+y)*{L});")
        A(f"      for (int z = 0; z < {L}; z++) {{ p[2*z] = ar[(size_t)z*{L}]; p[2*z+1] = ai[(size_t)z*{L}]; }}")
        A(f"    }} }}")
        A(f"}}")
    # buffers
    A(f"static double *S{L}r, *S{L}i, *T{L}r, *T{L}i, *cA{L}r, *cA{L}i, *cB{L}r, *cB{L}i;")
    if big:
        A(f"static double *W{L}r, *W{L}i;")
    else:
        A(f"static double *cC{L}r, *cC{L}i;")
    if L == 45:
        A(f"static void run_core_{L}(long Bc, long m, const double* x0r, const double* x0i, const double* ccr, const double* cci, double* out1, double* outm){{")
        A(f"  for (long b = 0; b < Bc; b++) {{")
        A(f"    const double* xbr = x0r + (size_t)b*{L**3}; const double* xbi = x0i + (size_t)b*{L**3};")
        A(f"    const double* cbr = ccr + (size_t)b*{L**3}; const double* cbi = cci + (size_t)b*{L**3};")
        A(f"    load_{L}(xbr, xbi, 1.0, S{L}r, S{L}i);")
        A(f"    if (m >= 2) load_{L}(cbr, cbi, 0.1, cA{L}r, cA{L}i);")
        A(f"    loadB_{L}(cbr, cbi, 0.1, cB{L}r, cB{L}i);")
        A(f"    double* sr = S{L}r; double* si = S{L}i;")
        A(f"    if (m < 1) {{ emit_{L}(sr, si, 0, out1 + 2*(size_t)b*{L**3}); emit_{L}(sr, si, 0, outm + 2*(size_t)b*{L**3}); continue; }}")
        A(f"    for (int g = 0; g < 9; g++) grp_planes1_45(sr, si, W{L}r, W{L}i, g, 1);")
        A(f"    for (long t = 1; t <= m; t++) {{")
        A(f"      int odd = (int)(t & 1);")
        A(f"      const double* ur = odd ? cB{L}r : cA{L}r;")
        A(f"      const double* ui = odd ? cB{L}i : cA{L}i;")
        A(f"      int n2g = odd ? 5 : 9;")
        A(f"      for (int k1 = 0; k1 < n2g; k1++) {{")
        A(f"        grp_stage2_45(sr, si, ur, ui, k1, odd);")
        A(f"        if (t == 1 && m > 1) {{")
        A(f"          for (int k2 = 0; k2 < 9; k2++) {{ int q = 9*k1 + k2; emitp1_45(sr, si, q, out1 + 2*(size_t)b*{L**3}); }}")
        A(f"        }}")
        A(f"        if (t < m) grp_planes1_45(sr, si, W{L}r, W{L}i, k1, !odd);")
        A(f"      }}")
        A(f"    }}")
        A(f"    if (m == 1) {{ emit_{L}(sr, si, 1, out1 + 2*(size_t)b*{L**3}); }}")
        A(f"    emit_{L}(sr, si, (int)(m & 1), outm + 2*(size_t)b*{L**3});")
        A(f"  }}")
        A(f"}}")
        A(f"void run_{L}(long Bc, long m, const double* x0r, const double* x0i, const double* ccr, const double* cci, double* out1, double* outm){{")
        A(f"  run_core_{L}(Bc, m, x0r, x0i, ccr, cci, out1, outm);")
        A(f"}}")
        A(f"void rungen_{L}(long Bc, long m, const uint64_t* wx, const uint64_t* wc, double* out1, double* outm){{")
        A(f"  size_t n = (size_t)Bc * {L**3};")
        A(f"  gens_ensure(n);")
        A(f"  fill_normals2(wx, 2*(long)n, Gxr, wc, 2*(long)n, Gcr);")
        A(f"  run_core_{L}(Bc, m, Gxr, Gxr + n, Gcr, Gcr + n, out1, outm);")
        A(f"}}")
        return "\n".join(s)
    if L in (36, 64):
        r0 = {36: 6, 64: 8}[L]
        if L == 64:
            ploop = f"for (int y = 0; y < 64; y++) for (int zc = 0; zc < 64; zc += 8) {{ int p = y*{ZP} + zc;"
        else:
            ploop = f"for (int p = 0; p < {L*ZP}; p += 8) {{"
        # helpers: planes+stage1 for a group, stage2 for a group
        A(f"static void grp_planes1_{L}(double* sr, double* si, double* wr, double* wi, int g, int odd){{")
        A(f"  for (int a = 0; a < {r0}; a++) {{")
        A(f"    size_t o = odd ? (size_t)({r0}*a + g)*{XS} : (size_t)({r0}*g + a)*{XS};")
        A(f"    ayp_{L}(sr + o, si + o, wr, wi);")
        A(f"    azp_{L}(wr, wi, sr + o, si + o);")
        A(f"  }}")
        A(f"  if (odd) switch (g) {{")
        for g in range(r0):
            A(f"    case {g}: {ploop} ax1_{L}_strided_{g}(sr + (size_t){g}*{XS} + p, si + (size_t){g}*{XS} + p); }} break;")
        A(f"  }} else switch (g) {{")
        for g in range(r0):
            A(f"    case {g}: {ploop} ax1_{L}_consec_{g}(sr + (size_t){g}*{r0*XS} + p, si + (size_t){g}*{r0*XS} + p); }} break;")
        A(f"  }}")
        A(f"}}")
        A(f"static void grp_stage2_{L}(double* sr, double* si, const double* cr, const double* ci, int k1, int odd){{")
        A(f"  if (odd) {{ {ploop} size_t o = (size_t)k1*{r0*XS} + p; ax2_{L}_consec(sr + o, si + o, cr + o, ci + o); }} }}")
        A(f"  else {{ {ploop} size_t o = (size_t)k1*{XS} + p; ax2_{L}_strided(sr + o, si + o, cr + o, ci + o); }} }}")
        A(f"}}")
        A(f"static void run_core_{L}(long Bc, long m, const double* x0r, const double* x0i, const double* ccr, const double* cci, double* out1, double* outm){{")
        A(f"  for (long b = 0; b < Bc; b++) {{")
        A(f"    const double* xbr = x0r + (size_t)b*{L**3}; const double* xbi = x0i + (size_t)b*{L**3};")
        A(f"    const double* cbr = ccr + (size_t)b*{L**3}; const double* cbi = cci + (size_t)b*{L**3};")
        A(f"    load_{L}(xbr, xbi, 1.0, S{L}r, S{L}i);")
        A(f"    if (m >= 2) load_{L}(cbr, cbi, 0.1, cA{L}r, cA{L}i);")
        A(f"    loadB_{L}(cbr, cbi, 0.1, cB{L}r, cB{L}i);")
        A(f"    double* sr = S{L}r; double* si = S{L}i;")
        A(f"    if (m < 1) {{ emit_{L}(sr, si, 0, out1 + 2*(size_t)b*{L**3}); emit_{L}(sr, si, 0, outm + 2*(size_t)b*{L**3}); continue; }}")
        A(f"    // step 1 planes+stage1 (odd)")
        A(f"    for (int g = 0; g < {r0}; g++) grp_planes1_{L}(sr, si, W{L}r, W{L}i, g, 1);")
        A(f"    for (long t = 1; t <= m; t++) {{")
        A(f"      int odd = (int)(t & 1);")
        A(f"      const double* ur = odd ? cB{L}r : cA{L}r;")
        A(f"      const double* ui = odd ? cB{L}i : cA{L}i;")
        A(f"      for (int k1 = 0; k1 < {r0}; k1++) {{")
        A(f"        grp_stage2_{L}(sr, si, ur, ui, k1, odd);")
        A(f"        if (t == 1 && m > 1) {{")
        A(f"          // snapshot planes of this group (orientation 1)")
        A(f"          for (int k2 = 0; k2 < {r0}; k2++) {{ int q = odd ? ({r0}*k1 + k2) : (k1 + {r0}*k2); emitp1_{L}(sr, si, q, out1 + 2*(size_t)b*{L**3}); }}")
        A(f"        }}")
        A(f"        if (t < m) {{")
        A(f"          int g = k1; int nodd = !odd;")
        A(f"          grp_planes1_{L}(sr, si, W{L}r, W{L}i, g, nodd);")
        A(f"        }}")
        A(f"      }}")
        A(f"    }}")
        A(f"    if (m == 1) {{ emit_{L}(sr, si, 1, out1 + 2*(size_t)b*{L**3}); }}")
        A(f"    emit_{L}(sr, si, (int)(m & 1), outm + 2*(size_t)b*{L**3});")
        A(f"  }}")
        A(f"}}")
        A(f"void run_{L}(long Bc, long m, const double* x0r, const double* x0i, const double* ccr, const double* cci, double* out1, double* outm){{")
        A(f"  run_core_{L}(Bc, m, x0r, x0i, ccr, cci, out1, outm);")
        A(f"}}")
        A(f"void rungen_{L}(long Bc, long m, const uint64_t* wx, const uint64_t* wc, double* out1, double* outm){{")
        A(f"  size_t n = (size_t)Bc * {L**3};")
        A(f"  gens_ensure(n);")
        A(f"  fill_normals2(wx, 2*(long)n, Gxr, wc, 2*(long)n, Gcr);")
        A(f"  run_core_{L}(Bc, m, Gxr, Gxr + n, Gcr, Gcr + n, out1, outm);")
        A(f"}}")
        return "\n".join(s)
    A(f"static void run_core_{L}(long Bc, long m, const double* x0r, const double* x0i, const double* ccr, const double* cci, double* out1, double* outm){{")
    A(f"  for (long b = 0; b < Bc; b++) {{")
    A(f"    const double* xbr = x0r + (size_t)b*{L**3}; const double* xbi = x0i + (size_t)b*{L**3};")
    A(f"    const double* cbr = ccr + (size_t)b*{L**3}; const double* cbi = cci + (size_t)b*{L**3};")
    A(f"    load_{L}(xbr, xbi, 1.0, S{L}r, S{L}i);")
    if big:
        A(f"    if (m >= 2) load_{L}(cbr, cbi, 0.1, cA{L}r, cA{L}i);")
        A(f"    loadB_{L}(cbr, cbi, 0.1, cB{L}r, cB{L}i);")
    else:
        A(f"    if (m >= 3) load_{L}(cbr, cbi, 0.1, cA{L}r, cA{L}i);")
        A(f"    loadB_{L}(cbr, cbi, 0.1, cB{L}r, cB{L}i);")
        A(f"    if (m >= 2) loadC_{L}(cbr, cbi, 0.1, cC{L}r, cC{L}i);")
    A(f"    double *cr_ = S{L}r, *ci_ = S{L}i, *tr = T{L}r, *ti = T{L}i;")
    A(f"    int orient = 0;")
    A(f"    if (m < 1) {{ emit_{L}(cr_, ci_, 0, out1 + 2*(size_t)b*{L**3}); emit_{L}(cr_, ci_, 0, outm + 2*(size_t)b*{L**3}); continue; }}")
    A(f"    for (long t = 1; t <= m; t++) {{")
    if big:
        A(f"      orient ^= 1;")
        A(f"      const double* ur = orient ? cB{L}r : cA{L}r;")
        A(f"      const double* ui = orient ? cB{L}i : cA{L}i;")
        if L in (36, 64):
            A(f"      step_{L}(cr_, ci_, W{L}r, W{L}i, ur, ui, orient);")
        elif L == 45:
            A(f"      step_{L}(cr_, ci_, W{L}r, W{L}i, tr, ti, ur, ui);")
        else:
            A(f"      step_{L}(cr_, ci_, W{L}r, W{L}i, ur, ui);")
    else:
        A(f"      orient = (orient + 1) % 3;")
        A(f"      const double* ur = orient == 1 ? cB{L}r : (orient == 2 ? cC{L}r : cA{L}r);")
        A(f"      const double* ui = orient == 1 ? cB{L}i : (orient == 2 ? cC{L}i : cA{L}i);")
        A(f"      step_{L}(cr_, ci_, tr, ti, ur, ui);")
        A(f"      {{ double* q; q = cr_; cr_ = tr; tr = q; q = ci_; ci_ = ti; ti = q; }}")
    A(f"      if (t == 1) emit_{L}(cr_, ci_, orient, out1 + 2*(size_t)b*{L**3});")
    A(f"    }}")
    A(f"    emit_{L}(cr_, ci_, orient, outm + 2*(size_t)b*{L**3});")
    A(f"  }}")
    A(f"}}")
    A(f"void run_{L}(long Bc, long m, const double* x0r, const double* x0i, const double* ccr, const double* cci, double* out1, double* outm){{")
    A(f"  run_core_{L}(Bc, m, x0r, x0i, ccr, cci, out1, outm);")
    A(f"}}")
    A(f"void rungen_{L}(long Bc, long m, const uint64_t* wx, const uint64_t* wc, double* out1, double* outm){{")
    A(f"  size_t n = (size_t)Bc * {L**3};")
    A(f"  gens_ensure(n);")
    A(f"  fill_normals2(wx, 2*(long)n, Gxr, wc, 2*(long)n, Gcr);")
    A(f"  run_core_{L}(Bc, m, Gxr, Gxr + n, Gcr, Gcr + n, out1, outm);")
    A(f"}}")
    return "\n".join(s)

def alloc_code():
    s = []
    A = lambda t: s.append(t)
    A('''static char* slab_base; static size_t slab_off, slab_cap;
static double* amem(size_t n){
  size_t bytes = ((n + 64) * sizeof(double) + 4095) & ~(size_t)4095;
  double* p = (double*)(slab_base + slab_off);
  slab_off += bytes;
  return p;
}''')
    A("void setup(void){")
    tot = 0
    per = lambda n: (((n + 64) * 8 + 4095) & ~4095)
    for L in SIZES:
        XS = XSMAP[L]
        n = L * XS
        cnt = 8 if L in BIG else 10
        tot += per(n) * cnt
        if L in BIG:
            tot += 2 * per(ZPMAP[L]**2)
    tot = (tot + (2<<20) - 1) & ~((2<<20)-1)
    A(f"  slab_cap = {tot}ULL + (2ULL<<20);")
    A("  void* raw = mmap(0, slab_cap + (2<<20), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);")
    A("  uintptr_t a = ((uintptr_t)raw + ((2<<20)-1)) & ~(uintptr_t)((2<<20)-1);")
    A("  slab_base = (char*)a; slab_off = 0;")
    A("  madvise(slab_base, slab_cap, MADV_HUGEPAGE);")
    A("  memset(slab_base, 0, slab_cap);")
    A("  init_wi_signed();")
    for L in SIZES:
        XS = XSMAP[L]
        n = L * XS
        A(f"  S{L}r = amem({n}); S{L}i = amem({n}); T{L}r = amem({n}); T{L}i = amem({n});")
        A(f"  cA{L}r = amem({n}); cA{L}i = amem({n}); cB{L}r = amem({n}); cB{L}i = amem({n});")
        if L in BIG:
            zp = ZPMAP[L]
            A(f"  W{L}r = amem({zp*zp}); W{L}i = amem({zp*zp});")
        else:
            A(f"  cC{L}r = amem({n}); cC{L}i = amem({n});")
    A("}")
    return "\n".join(s)

def main():
    parts = [HEADER]
    STAGED = {36: ('pfa',4,9), 45: ('pfa',5,9), 64: ('ct',8,8)}
    RADER_MB = {13: 5, 17: 5, 23: 7}
    for L in SIZES:
        XS = XSMAP[L]
        if L in STAGED:
            plain = gen.emit_codelet_staged(L, 8, STAGED[L])
            fmap  = gen.emit_codelet_staged(L, 8, STAGED[L], mapstore=True)
        elif L in RADER_MB:
            plain = gen.emit_rader(L, 8, maxblock=RADER_MB[L])
            fmap  = gen.emit_rader(L, 8, mapstore=True, maxblock=RADER_MB[L])
        else:
            plain = gen.emit_codelet(L, 8)
            fmap  = emit_map_codelet(L, 8)
        if L in BIG:
            zp = ZPMAP[L]
            parts.append(specialize(plain, f"fft{L}_w8", f"fft{L}_yy", zp, zp))     # ayp
            parts.append(specialize(plain, f"fft{L}_w8", f"fft{L}_zz", 8, zp))      # azp
            if L in (36, 64):
                r = {36: 6, 64: 8}[L]
                for shape in ('strided', 'consec'):
                    for g in range(r):
                        parts.append(gen.emit_ax_stage1(r, XS, shape, g, f"ax1_{L}_{shape}_{g}"))
                    parts.append(gen.emit_ax_stage2map(r, XS, shape, f"ax2_{L}_{shape}"))
            elif L == 45:
                for b in range(9):
                    parts.append(gen.emit_ax45_stage1(5, 9, XS, b, f"ax1_45_s19_{b}", consec=False))
                for b in range(5):
                    parts.append(gen.emit_ax45_stage1(9, 5, XS, b, f"ax1_45_s95_{b}", consec=True))
                parts.append(gen.emit_ax45_stage2map(9, XS, "ax2_45_s19", consec=True))
                parts.append(gen.emit_ax45_stage2map(5, XS, "ax2_45_s95", consec=False))
            else:
                parts.append(specialize(fmap, f"fftmap{L}_w8", f"fftmap{L}_pp", XS, XS))
        else:
            parts.append(specialize(plain, f"fft{L}_w8", f"fft{L}_yy", L, L))       # ay
            parts.append(specialize(plain, f"fft{L}_w8", f"fft{L}_pp", XS, XS))     # ax
            parts.append(specialize(fmap,  f"fftmap{L}_w8", f"fftmap{L}_zz", 8, XS).replace("MAPCALL(", "map8nr(")) # az rotation
        if L == 6:
            parts.append(specialize(gen.emit_codelet(L, 4), "fft6_w4", "fft6_yy4", L, L))

    for L in SIZES:
        parts.append(drv_for(L))
        parts.append(io_for(L))
    parts.append(alloc_code())
    src = "\n\n".join(parts)
    open(sys.argv[1] if len(sys.argv) > 1 else 'implementation.c', 'w').write(src)
    print(f"wrote {len(src)} bytes")

if __name__ == '__main__':
    main()
