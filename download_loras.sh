#!/bin/bash
cd /models/loras || exit 1

# Function to download a file with proper URL encoding
download_file() {
  filename="$1"
  # URL encode the filename for the URL
  encoded_url=$(echo "$filename" | sed -e "s/ /%20/g")
  echo "Downloading $filename..."
  wget -q -O "$filename" "https://d1s3da0dcaf6kx.cloudfront.net/$encoded_url" || {
    echo "Failed to download $filename"
    return 1
  }
  echo "Successfully downloaded $filename"
  return 0
}

# Download all LoRA files
download_file "wan-nsfw-e14-fixed.safetensors"
download_file "big_tits_epoch_50.safetensors"
download_file "pov_blowjob_v1.1.safetensors"
download_file "Wan_Breast_Helper_Hearmeman.safetensors"
download_file "wan_cowgirl_v1.3.safetensors"
download_file "cleavage_epoch_40.safetensors"
download_file "orgasm_e60.safetensors"
download_file "wan_missionary_side.safetensors"
download_file "dicks_epoch_100.safetensors"
download_file "masturbation_cumshot_wanI2V480p_v1.safetensors"
download_file "r0und4b0ut-wan-v1.0.safetensors"
download_file "facials_epoch_50.safetensors"
download_file "deepthroat_epoch_80.safetensors"
download_file "ahegao_v1_e35_wan.safetensors"
download_file "Wan_Pussy_LoRA_Hearmeman.safetensors"
download_file "doggyPOV_v1_1.safetensors"
download_file "wan_pov_missionary_v1.1.safetensors"
download_file "Titfuck_WAN14B_V1_Release.safetensors"
download_file "FILM_NOIR_EPOCH10.safetensors"
download_file "BouncyWalkV01.safetensors"
download_file "Spinning V2.safetensors"
download_file "squish_18.safetensors"
download_file "detailz-wan.safetensors"
download_file "studio_ghibli_wan14b_t2v_v01.safetensors"
download_file "Su_Bl_Ep02-Wan.safetensors"
download_file "wan_female_masturbation.safetensors"
download_file "Wan-Hip_Slammin_Assertive_Cowgirl.safetensors"
download_file "T2V - Skinny Petite Instagram Women - 14B.safetensors"
download_file "T2V-jiggle_tits-14b.safetensors"
download_file "PPD_v-9.safetensors"
download_file "DR34M15H_T2V_HD_v1.safetensors"
download_file "T2V - Big Perky Breasts - 14b.safetensors"
download_file "T2V - Bodyshots v2 - 14b.safetensors"
download_file "SECRET_SAUCE_WAN2.1_14B_fp8.safetensors"
download_file "BoobPhysics_WAN_v7.safetensors"
download_file "T2V - Dildo Masturbation - 14B.safetensors"
download_file "sh4rpn3ss_v2_e56.safetensors"


echo "All LoRA files downloaded successfully"