"""
Master script to generate all distortion types.

This script runs all individual distortion generation scripts in sequence.

Usage:
    python generate_all_distortions.py --wav_scp clean_wav.scp --segments clean_segments \
                                       --output_dir output \
                                       --noise_wav_scp noise.scp \
                                       --rir_wav_scp rir.scp \
                                       --interference_wav_scp interference.scp
"""

import os
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*60)
    print(f"Running: {description}")
    print("="*60)
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nWarning: {description} failed with return code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate all distortion types from clean audio"
    )
    
    # Input/Output
    parser.add_argument('--wav_scp', type=str, required=True,
                       help='Kaldi wav.scp file for clean audio')
    parser.add_argument('--segments', type=str, required=True,
                       help='Kaldi segments file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Root output directory')
    
    # Resource files (optional - if not provided, that distortion type will be skipped)
    parser.add_argument('--noise_wav_scp', type=str, default=None,
                       help='wav.scp file for noise resources')
    parser.add_argument('--noise_segments', type=str, default=None,
                       help='segments file for noise resources')
    parser.add_argument('--rir_wav_scp', type=str, default=None,
                       help='wav.scp file for RIR resources')
    parser.add_argument('--rir_segments', type=str, default=None,
                       help='segments file for RIR resources')
    parser.add_argument('--interference_wav_scp', type=str, default=None,
                       help='wav.scp file for interference resources')
    parser.add_argument('--interference_segments', type=str, default=None,
                       help='segments file for interference resources')
    
    # Parameters
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Target sample rate (default: 16000)')
    parser.add_argument('--frame_hop_ms', type=float, default=20.0,
                       help='Frame hop in milliseconds (default: 20 for 50fps)')
    parser.add_argument('--snr_range', type=float, nargs=2, default=[-5, 20],
                       help='SNR range for noise (default: -5 20)')
    parser.add_argument('--sir_range', type=float, nargs=2, default=[-5, 10],
                       help='SIR range for interference (default: -5 10)')
    parser.add_argument('--packet_loss_bitrates', type=int, nargs='+',
                       default=[6, 8, 12, 16],
                       help='Bitrates for packet loss (default: 6 8 12 16)')
    parser.add_argument('--max_segments', type=int, default=None,
                       help='Maximum number of segments to process (for testing)')
    
    # Selection
    parser.add_argument('--skip', type=str, nargs='+', default=[],
                       choices=['clean', 'noise', 'rir', 'interference', 'packet_loss', 'missing'],
                       help='Distortion types to skip')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_cmd_args = [
        '--sample_rate', str(args.sample_rate),
        '--frame_hop_ms', str(args.frame_hop_ms),
    ]
    
    if args.max_segments:
        base_cmd_args.extend(['--max_segments', str(args.max_segments)])
    
    # Track which distortions were generated
    results = {}
    
    # 1. Clean
    if 'clean' not in args.skip:
        cmd = [
            'python', 'generate_clean.py',
            '--wav_scp', args.wav_scp,
            '--segments', args.segments,
            '--output_dir', str(output_dir / 'clean'),
        ] + base_cmd_args
        results['clean'] = run_command(cmd, "Clean audio generation")
    else:
        print("\nSkipping clean audio generation")
    
    # 2. Noise
    if 'noise' not in args.skip and args.noise_wav_scp:
        cmd = [
            'python', 'generate_noise.py',
            '--wav_scp', args.wav_scp,
            '--segments', args.segments,
            '--noise_wav_scp', args.noise_wav_scp,
            '--output_dir', str(output_dir / 'noise'),
            '--snr_range', str(args.snr_range[0]), str(args.snr_range[1]),
        ] + base_cmd_args
        
        if args.noise_segments:
            cmd.extend(['--noise_segments', args.noise_segments])
        
        results['noise'] = run_command(cmd, "Noisy audio generation")
    else:
        print("\nSkipping noise generation (no noise resources or explicitly skipped)")
    
    # 3. RIR
    if 'rir' not in args.skip and args.rir_wav_scp:
        cmd = [
            'python', 'generate_rir.py',
            '--wav_scp', args.wav_scp,
            '--segments', args.segments,
            '--rir_wav_scp', args.rir_wav_scp,
            '--output_dir', str(output_dir / 'rir'),
        ] + base_cmd_args
        
        if args.rir_segments:
            cmd.extend(['--rir_segments', args.rir_segments])
        
        results['rir'] = run_command(cmd, "Reverberant audio generation")
    else:
        print("\nSkipping RIR generation (no RIR resources or explicitly skipped)")
    
    # 4. Interference
    if 'interference' not in args.skip and args.interference_wav_scp:
        cmd = [
            'python', 'generate_interference.py',
            '--wav_scp', args.wav_scp,
            '--segments', args.segments,
            '--interference_wav_scp', args.interference_wav_scp,
            '--output_dir', str(output_dir / 'interference'),
            '--sir_range', str(args.sir_range[0]), str(args.sir_range[1]),
        ] + base_cmd_args
        
        if args.interference_segments:
            cmd.extend(['--interference_segments', args.interference_segments])
        
        results['interference'] = run_command(cmd, "Interference audio generation")
    else:
        print("\nSkipping interference generation (no interference resources or explicitly skipped)")
    
    # 5. Packet Loss
    if 'packet_loss' not in args.skip:
        cmd = [
            'python', 'generate_packet_loss.py',
            '--wav_scp', args.wav_scp,
            '--segments', args.segments,
            '--output_dir', str(output_dir / 'packet_loss'),
            '--bitrates',
        ] + [str(b) for b in args.packet_loss_bitrates] + base_cmd_args
        
        results['packet_loss'] = run_command(cmd, "Packet loss audio generation")
    else:
        print("\nSkipping packet loss generation")
    
    # 6. Missing
    if 'missing' not in args.skip:
        cmd = [
            'python', 'generate_missing.py',
            '--wav_scp', args.wav_scp,
            '--segments', args.segments,
            '--output_dir', str(output_dir / 'missing'),
        ] + base_cmd_args
        
        results['missing'] = run_command(cmd, "Missing segment audio generation")
    else:
        print("\nSkipping missing segment generation")
    
    # Print final summary
    print("\n" + "="*60)
    print("ALL DISTORTIONS COMPLETE!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print("\nGeneration results:")
    for dist_type, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {dist_type:15s}: {status}")
    print("\nEach distortion type has its own subdirectory:")
    for subdir in ['clean', 'noise', 'rir', 'interference', 'packet_loss', 'missing']:
        if (output_dir / subdir).exists():
            num_files = len(list((output_dir / subdir).glob('*.wav')))
            print(f"  {subdir:15s}: {num_files} files")
    print("="*60)


if __name__ == "__main__":
    main()
